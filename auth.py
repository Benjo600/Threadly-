from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import jwt
import datetime
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.urandom(24)  # Change this to a secure secret key in production

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, app.secret_key, algorithm='HS256')

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    
    # Validate input
    if not all(k in data for k in ['name', 'email', 'password']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if user already exists
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE email = ?', (data['email'],))
    if c.fetchone():
        conn.close()
        return jsonify({'error': 'Email already registered'}), 400
    
    # Hash password and create user
    hashed_password = generate_password_hash(data['password'])
    try:
        c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                 (data['name'], data['email'], hashed_password))
        conn.commit()
        user_id = c.lastrowid
        token = generate_token(user_id)
        conn.close()
        return jsonify({
            'message': 'User created successfully',
            'token': token
        }), 201
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Validate input
    if not all(k in data for k in ['email', 'password']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check user credentials
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, password FROM users WHERE email = ?', (data['email'],))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], data['password']):
        token = generate_token(user[0])
        return jsonify({
            'message': 'Login successful',
            'token': token
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/verify-token', methods=['GET'])
def verify_token():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'No token provided'}), 401
    
    try:
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        return jsonify({'valid': True, 'user_id': payload['user_id']})
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401

if __name__ == '__main__':
    app.run(debug=True, port=5000) 