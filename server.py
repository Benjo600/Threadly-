from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
from groq import Groq
import os
import logging
import re
import requests
from dotenv import load_dotenv
import json
import time
from datetime import datetime
from yt_dlp import YoutubeDL
import threading

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variable validation
required_env_vars = {
    "GROQ_API_KEY": "API key for Groq service",
    "DAILY_TOKEN_LIMIT": "Daily token usage limit"
}

missing_vars = []
for var, description in required_env_vars.items():
    if not os.environ.get(var):
        missing_vars.append(f"{var} ({description})")

if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Initialize Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Default values for prompt placeholders
PERSONA_ROLE = "content creator"
INDUSTRY = "social media"
TARGET_AUDIENCE = "professionals"

# Token usage tracking configuration
TOKEN_USAGE_FILE = 'token_usage.json'
DAILY_TOKEN_LIMIT = int(os.environ["DAILY_TOKEN_LIMIT"])
TOKEN_USAGE_LOCK = threading.Lock()

def load_token_usage():
    try:
        with open(TOKEN_USAGE_FILE, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'recent_usages' not in data or 'daily_total' not in data or 'last_reset_date' not in data:
            raise ValueError("Invalid token usage data")
        return data
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {
            "recent_usages": [],
            "daily_total": 0,
            "last_reset_date": datetime.utcnow().strftime("%Y-%m-%d")
        }

def save_token_usage(data):
    with TOKEN_USAGE_LOCK:
        with open(TOKEN_USAGE_FILE, 'w') as f:
            json.dump(data, f)

def update_token_usage(tokens):
    with TOKEN_USAGE_LOCK:
        data = load_token_usage()
        current_time = time.time()
        current_date = datetime.utcnow().strftime("%Y-%m-%d")

        if current_date != data["last_reset_date"]:
            data["daily_total"] = 0
            data["last_reset_date"] = current_date

        data["recent_usages"].append({"timestamp": current_time, "tokens": tokens})
        data["daily_total"] += tokens

        cutoff = current_time - 300  # Keep last 5 minutes
        data["recent_usages"] = [usage for usage in data["recent_usages"] if usage["timestamp"] >= cutoff]

        save_token_usage(data)

def get_token_usage_stats():
    data = load_token_usage()
    current_time = time.time()
    cutoff = current_time - 60  # Last minute
    tokens_last_minute = sum(usage["tokens"] for usage in data["recent_usages"] if usage["timestamp"] >= cutoff)
    tokens_used_today = data["daily_total"]
    tokens_remaining_today = max(0, DAILY_TOKEN_LIMIT - tokens_used_today)
    return {
        "tokens_last_minute": tokens_last_minute,
        "tokens_used_today": tokens_used_today,
        "tokens_remaining_today": tokens_remaining_today
    }

def chunk_transcript(transcript, max_tokens=4000):
    words = transcript.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    avg_chars_per_token = 4

    for word in words:
        word_tokens = len(word) // avg_chars_per_token + 1
        if current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def clean_youtube_url(url):
    """Extract video ID from various YouTube URL formats."""
    try:
        patterns = [
            r'(?:v=|v/|embed/|youtu.be/)([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?(?:youtube\.com|youtu\.be)/(?:watch\?v=)?([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    except Exception as e:
        logger.error(f"URL cleaning failed: {str(e)}")
        return None

def get_transcript(yt_link):
    try:
        if not yt_link or not isinstance(yt_link, str):
            logger.error("Invalid YouTube link provided")
            return None, False

        video_id = clean_youtube_url(yt_link)
        logger.info(f"Extracted video ID: {video_id}")
        if not video_id:
            logger.error(f"Invalid YouTube URL or could not extract video ID: {yt_link}")
            return None, False

        clean_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Cleaned URL: {clean_url}")

        duration = None
        # Try yt-dlp first
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'socket_timeout': 10,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=False)
                duration = info.get('duration')
                logger.info(f"Video duration from yt-dlp: {duration} seconds")
        except Exception as e:
            logger.warning(f"yt-dlp failed to fetch duration: {str(e)}")

        # If yt-dlp fails, try pytube
        if duration is None:
            try:
                yt = YouTube(clean_url)
                yt.check_availability()  # Verify video is available
                duration = yt.length
                logger.info(f"Video duration from pytube: {duration} seconds")
            except Exception as e:
                logger.warning(f"Pytube failed to fetch duration: {str(e)}")

        if duration and duration > 1800:
            logger.warning(f"Video duration ({duration} seconds) exceeds 30 minutes limit")
            return None, True

        try:
            # First try to get available transcript languages
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            logger.info(f"Available transcript languages: {[t.language_code for t in transcript_list]}")
            
            # Try to get English transcript first
            try:
                transcript = transcript_list.find_transcript(['en'])
                transcript_data = transcript.fetch()
            except:
                # If English not available, try to get any available transcript
                transcript = transcript_list.find_transcript()
                transcript_data = transcript.fetch()
            
            if not transcript_data:
                logger.error("Empty transcript data received")
                return None, False
                
            # Process the transcript data correctly
            text_parts = []
            for entry in transcript_data:
                if hasattr(entry, 'text'):
                    text_parts.append(entry.text)
                elif isinstance(entry, dict) and 'text' in entry:
                    text_parts.append(entry['text'])
                else:
                    logger.warning(f"Unexpected transcript entry format: {entry}")
                    continue
                    
            text = ' '.join(text_parts)
            if not text.strip():
                logger.error("Empty transcript text")
                return None, False
                
            logger.info(f"Successfully extracted transcript for video {video_id}")
            return text, False
            
        except TranscriptsDisabled:
            logger.error(f"Transcripts are disabled for video {video_id}")
            return None, False
        except NoTranscriptFound:
            logger.error(f"No transcript found for video {video_id}")
            return None, False
        except Exception as e:
            logger.error(f"Error fetching transcript: {str(e)}")
            return None, False
            
    except Exception as e:
        logger.error(f"Transcript extraction failed: {str(e)}")
        return None, False

def generate_posts(transcript, platform, num_posts, custom_instructions, custom_cta, tone, pain_points, ai_model):
    try:
        logger.info(f"Starting post generation with parameters: platform={platform}, num_posts={num_posts}, tone={tone}")
        chunks = chunk_transcript(transcript, max_tokens=4000)
        logger.info(f"Split transcript into {len(chunks)} chunks")
        all_posts = []
        total_tokens = 0

        posts_per_chunk = max(1, num_posts // len(chunks) + (1 if num_posts % len(chunks) else 0))
        logger.info(f"Generating {posts_per_chunk} posts per chunk")

        # Select platform-specific prompt instructions
        if platform.lower() == 'linkedin':
            system_message = f"You are a {PERSONA_ROLE}, a {INDUSTRY} expert with over 1 million LinkedIn followers, known for sharing actionable insights."
            post_instructions = """
1. Create a bold opener that challenges a common belief held by {target_audience} to grab attention and compel them to read the full post.
2. Address the audience's {pain_points} concisely in 1-2 sentences, speaking directly to their struggles.
3. Agitate the pain points by highlighting the negative consequences of not addressing the problem, using an urgent yet professional tone.
4. Present a clear solution to the problem, drawing key insights from the YouTube transcript, and end with the provided call-to-action: {custom_cta}.
5. Ensure the post is professional, engaging, and aligns with LinkedIn's tone, with a maximum of 300 words. Extract only the most relevant points from the transcript to fit the post's purpose and audience.
""".format(
                target_audience=TARGET_AUDIENCE,
                pain_points=pain_points or 'struggling to make content',
                custom_cta=custom_cta or ''
            )
        elif platform.lower() == 'twitter':
            system_message = f"You are a {PERSONA_ROLE}, a {INDUSTRY} thought leader with a massive X following, known for bold, concise insights that spark conversation."
            post_instructions = """
Each post should follow this framework:
1. **Provocative Opener (1-2 sentences)**: Challenge a core belief of {target_audience} with a shocking or contrarian statement. Use bold phrasing (e.g., "Think [common belief]? You're dead wrong.") to stop scrolls and drive curiosity.
2. **Pain Point (1 sentence)**: Pinpoint one specific {pain_points} in a direct, relatable way, mirroring the audience's language (e.g., "Struggling to [issue]?").
3. **Agitation (1 sentence)**: Amplify the pain with a vivid consequence of inaction, using urgency or stakes (e.g., "Keep this up, and you'll lose [specific outcome].").
4. **Solution + CTA (1-2 sentences)**: Share a single, actionable insight from the transcript's key points, tailored to solve the pain. End with a clear, compelling call-to-action: {custom_cta} (e.g., "Watch now: [link]"). Avoid generic phrases like "learn more."
5. **Execution Guidelines**:
   - **Tone**: Bold, conversational, or witty, matching X's vibe. Avoid corporate jargon or buzzwords (e.g., "game-changer," "disruptor").
   - **Brevity**: Each post must be 280 characters or less, including spaces, links, and hashtags. Prioritize punchy words and short sentences.
   - **Transcript Handling**: Extract 1-2 key insights from the transcript most relevant to {target_audience}. Summarize or rephrase; don't quote directly unless it's under 50 characters.
   - **Engagement Boosters**: Include 1-2 relevant hashtags (e.g., #IndustryTrend, #AudienceTopic) and one emoji (e.g., 🚀, 💡) for visibility, but don't overdo it. Optionally mention a relevant handle (e.g., @IndustryLeader) if it fits.
   - **Quality Check**: Ensure each post feels fresh, avoids clichés, and aligns with {persona_role}'s expertise. Test for clarity by reading as a {target_audience} member.
""".format(
                target_audience=TARGET_AUDIENCE,
                pain_points=pain_points or 'struggling to make content',
                custom_cta=custom_cta or '',
                persona_role=PERSONA_ROLE
            )
        else:
            logger.error(f"Unsupported platform: {platform}")
            return None, None

        for chunk in chunks:
            logger.info(f"Processing chunk of length {len(chunk)}")
            prompt = f"""
Based on the following YouTube transcript chunk, create {posts_per_chunk} unique {platform} posts following these instructions:

{post_instructions}

Each post must be distinct, avoiding repetition of content from other posts.
Use a {tone.lower()} tone and directly address the audience pain point: '{pain_points or 'struggling to make content'}'.
Follow these custom instructions strictly: '{custom_instructions or 'focus on actionable strategies'}'.
Output ONLY the posts themselves, with no introductory text, explanations, or statements like 'Here are the posts'.
Separate each post with exactly two newlines (\n\n).

Transcript chunk: {chunk}
"""
            logger.info("Sending request to Groq API")
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Correct model name for Groq
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,  # Increased from 1500 to allow for more content
                    temperature=0.7,  # Reduced from 1.0 for more consistent outputs
                    top_p=0.9,  # Added to improve quality
                    frequency_penalty=0.5,  # Added to reduce repetition
                    presence_penalty=0.5  # Added to encourage diverse content
                )
                logger.info("Successfully received response from Groq API")
                
                # Debug log the entire response structure
                logger.info(f"Response structure: {dir(response)}")
                logger.info(f"Response choices: {response.choices}")
                
                if not response.choices:
                    logger.error("No choices in response")
                    continue
                    
                if not hasattr(response.choices[0], 'message'):
                    logger.error("No message in first choice")
                    continue
                    
                if not hasattr(response.choices[0].message, 'content'):
                    logger.error("No content in message")
                    continue
                
                content = response.choices[0].message.content
                if not content:
                    logger.error("Empty content in response")
                    continue
                    
                content = content.strip()
                logger.info(f"Raw API response content length: {len(content)}")
                logger.info(f"Raw API response content: {content[:200]}...")  # Log first 200 chars
                
                # Split content into posts, handling different possible separators
                posts = [p.strip() for p in content.split('\n\n') if p.strip()]
                logger.info(f"Split into {len(posts)} raw posts")
                
                # Filter out posts with unwanted preview text
                filtered_posts = [
                    post for post in posts
                    if post and not any(phrase in post.lower() for phrase in ['here are', 'based on the transcript', 'following posts'])
                ]
                logger.info(f"Filtered down to {len(filtered_posts)} valid posts")
                
                if filtered_posts:
                    all_posts.extend(filtered_posts)
                    total_tokens += response.usage.total_tokens if response.usage else 0
                    logger.info(f"Added {len(filtered_posts)} posts to final list")
                else:
                    logger.warning("No valid posts found in this chunk after filtering")
                    
            except Exception as e:
                logger.error(f"Error during API call or post processing: {str(e)}")
                logger.exception("Full traceback:")
                continue

        # Ensure we return exactly num_posts, or fewer if not enough valid posts
        all_posts = all_posts[:num_posts]
        if not all_posts:
            logger.error("No valid posts generated after processing all chunks")
            return None, None
            
        logger.info(f"Successfully generated {len(all_posts)} posts")
        token_usage = {"total_tokens": total_tokens}
        final_posts = '\n\n'.join(all_posts)
        logger.info(f"Final posts length: {len(final_posts)}")
        logger.info(f"First post preview: {final_posts[:200]}...")  # Log first 200 chars of first post
        return final_posts, token_usage
    except Exception as e:
        logger.error(f"Post generation failed with error: {str(e)}")
        logger.exception("Full traceback:")
        return None, None

@app.route('/', methods=['GET'])
def health_check():
    return "OK", 200

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        logger.info(f"Received summarize request with data: {data}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        # Required fields validation
        required_fields = ['yt_link']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        yt_link = data.get('yt_link')
        platform = data.get('platform', 'linkedin')
        num_posts = data.get('num_posts', 3)
        custom_instructions = data.get('custom_instructions', '')
        custom_cta = data.get('custom_cta', '')
        tone = data.get('tone', 'Professional')
        pain_points = data.get('pain_points', '')
        ai_model = data.get('ai_model', 'llama3-70b-8192')

        # Input validation
        if not isinstance(yt_link, str) or not yt_link.strip():
            return jsonify({"error": "Invalid YouTube link"}), 400

        if platform.lower() not in ['linkedin', 'twitter']:
            return jsonify({"error": "Unsupported platform. Supported platforms are: linkedin, twitter"}), 400

        if not isinstance(num_posts, int) or num_posts < 1 or num_posts > 10:
            return jsonify({"error": "Number of posts must be between 1 and 10"}), 400

        if not isinstance(tone, str) or not tone.strip():
            return jsonify({"error": "Invalid tone parameter"}), 400

        # Check token usage before proceeding
        token_stats = get_token_usage_stats()
        if token_stats["tokens_remaining_today"] <= 0:
            return jsonify({"error": "Daily token limit exceeded"}), 429

        transcript, upgrade_required = get_transcript(yt_link)
        if upgrade_required:
            return jsonify({"upgrade_required": True, "error": "Video exceeds 30 minutes"}), 403
        if not transcript:
            return jsonify({"error": "Unable to process video. It may lack captions, be restricted, or have metadata issues."}), 400

        posts, token_usage = generate_posts(transcript, platform, num_posts, custom_instructions, custom_cta, tone, pain_points, ai_model)
        if not posts:
            logger.error("No posts generated")
            return jsonify({"error": "Failed to generate posts"}), 500

        logger.info(f"Successfully generated {len(posts.split('\n\n'))} posts")
        response_data = {
            "summary": posts,
            "token_usage": token_usage,
            "video_id": clean_youtube_url(yt_link)
        }
        logger.info("Sending response to client")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Summarize endpoint failed: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/token_usage', methods=['GET'])
def token_usage():
    stats = get_token_usage_stats()
    return jsonify(stats)

@app.route('/repurpose', methods=['POST'])
def repurpose():
    try:
        data = request.get_json()
        logger.info(f"Received repurpose request with data: {data}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        # Required fields validation
        required_fields = ['text']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        input_text = data.get('text')
        platform = data.get('platform', 'linkedin')
        num_posts = data.get('num_posts', 3)
        custom_instructions = data.get('custom_instructions', '')
        custom_cta = data.get('custom_cta', '')
        tone = data.get('tone', 'Professional')
        pain_points = data.get('pain_points', '')

        # Input validation
        if not isinstance(input_text, str) or not input_text.strip():
            return jsonify({"error": "Invalid input text"}), 400

        if platform.lower() not in ['linkedin', 'twitter']:
            return jsonify({"error": "Unsupported platform. Supported platforms are: linkedin, twitter"}), 400

        if not isinstance(num_posts, int) or num_posts < 1 or num_posts > 10:
            return jsonify({"error": "Number of posts must be between 1 and 10"}), 400

        if not isinstance(tone, str) or not tone.strip():
            return jsonify({"error": "Invalid tone parameter"}), 400

        # Check token usage before proceeding
        token_stats = get_token_usage_stats()
        if token_stats["tokens_remaining_today"] <= 0:
            return jsonify({"error": "Daily token limit exceeded"}), 429

        # Select platform-specific prompt instructions
        if platform.lower() == 'linkedin':
            system_message = f"You are a {PERSONA_ROLE}, a {INDUSTRY} expert with over 1 million LinkedIn followers, known for sharing actionable insights."
            post_instructions = """
1. Create a bold opener that challenges a common belief held by {target_audience} to grab attention and compel them to read the full post.
2. Address the audience's {pain_points} concisely in 1-2 sentences, speaking directly to their struggles.
3. Agitate the pain points by highlighting the negative consequences of not addressing the problem, using an urgent yet professional tone.
4. Present a clear solution to the problem, drawing key insights from the provided text, and end with the provided call-to-action: {custom_cta}.
5. Ensure the post is professional, engaging, and aligns with LinkedIn's tone, with a maximum of 300 words. Extract only the most relevant points from the text to fit the post's purpose and audience.
""".format(
                target_audience=TARGET_AUDIENCE,
                pain_points=pain_points or 'struggling to make content',
                custom_cta=custom_cta or ''
            )
        else:  # twitter
            system_message = f"You are a {PERSONA_ROLE}, a {INDUSTRY} thought leader with a massive X following, known for bold, concise insights that spark conversation."
            post_instructions = """
Each post should follow this framework:
1. **Provocative Opener (1-2 sentences)**: Challenge a core belief of {target_audience} with a shocking or contrarian statement. Use bold phrasing (e.g., "Think [common belief]? You're dead wrong.") to stop scrolls and drive curiosity.
2. **Pain Point (1 sentence)**: Pinpoint one specific {pain_points} in a direct, relatable way, mirroring the audience's language (e.g., "Struggling to [issue]?").
3. **Agitation (1 sentence)**: Amplify the pain with a vivid consequence of inaction, using urgency or stakes (e.g., "Keep this up, and you'll lose [specific outcome].").
4. **Solution + CTA (1-2 sentences)**: Share a single, actionable insight from the text's key points, tailored to solve the pain. End with a clear, compelling call-to-action: {custom_cta} (e.g., "Watch now: [link]"). Avoid generic phrases like "learn more."
5. **Execution Guidelines**:
   - **Tone**: Bold, conversational, or witty, matching X's vibe. Avoid corporate jargon or buzzwords (e.g., "game-changer," "disruptor").
   - **Brevity**: Each post must be 280 characters or less, including spaces, links, and hashtags. Prioritize punchy words and short sentences.
   - **Text Handling**: Extract 1-2 key insights from the text most relevant to {target_audience}. Summarize or rephrase; don't quote directly unless it's under 50 characters.
   - **Engagement Boosters**: Include 1-2 relevant hashtags (e.g., #IndustryTrend, #AudienceTopic) and one emoji (e.g., 🚀, 💡) for visibility, but don't overdo it. Optionally mention a relevant handle (e.g., @IndustryLeader) if it fits.
   - **Quality Check**: Ensure each post feels fresh, avoids clichés, and aligns with {persona_role}'s expertise. Test for clarity by reading as a {target_audience} member.
""".format(
                target_audience=TARGET_AUDIENCE,
                pain_points=pain_points or 'struggling to make content',
                custom_cta=custom_cta or '',
                persona_role=PERSONA_ROLE
            )

        prompt = f"""
Based on the following text, create {num_posts} unique {platform} posts following these instructions:

{post_instructions}

Each post must be distinct, avoiding repetition of content from other posts.
Use a {tone.lower()} tone and directly address the audience pain point: '{pain_points or 'struggling to make content'}'.
Follow these custom instructions strictly: '{custom_instructions or 'focus on actionable strategies'}'.
Output ONLY the posts themselves, with no introductory text, explanations, or statements like 'Here are the posts'.
Separate each post with exactly two newlines (\n\n).

Text: {input_text}
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )

            if not response.choices or not response.choices[0].message.content:
                return jsonify({"error": "Failed to generate posts"}), 500

            content = response.choices[0].message.content.strip()
            posts = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Filter out posts with unwanted preview text
            filtered_posts = [
                post for post in posts
                if post and not any(phrase in post.lower() for phrase in ['here are', 'based on the text', 'following posts'])
            ]

            if not filtered_posts:
                return jsonify({"error": "No valid posts generated"}), 500

            # Ensure we return exactly num_posts, or fewer if not enough valid posts
            filtered_posts = filtered_posts[:num_posts]
            final_posts = '\n\n'.join(filtered_posts)

            # Update token usage
            if response.usage:
                update_token_usage(response.usage.total_tokens)

            return jsonify({
                "summary": final_posts,
                "token_usage": {"total_tokens": response.usage.total_tokens if response.usage else 0}
            })

        except Exception as e:
            logger.error(f"Error during API call or post processing: {str(e)}")
            logger.exception("Full traceback:")
            return jsonify({"error": "Failed to generate posts"}), 500

    except Exception as e:
        logger.error(f"Repurpose endpoint failed: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    app.run(debug=debug, port=5000)