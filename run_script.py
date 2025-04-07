from fastapi import FastAPI, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import subprocess
import mysql.connector
import bcrypt

app = FastAPI()

# Load HTML templates
templates = Jinja2Templates(directory="templates")

#POWER_BI_DASHBOARD_URL = "https://app.powerbi.com/groups/me/reports/6a9b55d1-c1df-40bc-8805-a4076f19e1ac/7902801c38c95ad39f5a?ctid=2d78bb54-b810-411b-b716-d0e5a53f10a8&experience=power-bi"
#POWER_BI_DASHBOARD_URL = "https://app.powerbi.com/links/zu4K47Vcyb?ctid=2d78bb54-b810-411b-b716-d0e5a53f10a8&pbi_source=linkShare"

POWER_BI_DASHBOARD_URL = "https://app.powerbi.com/groups/428fafb9-3e88-4a23-b980-c1e4e0732ac2/reports/6a9b55d1-c1df-40bc-8805-a4076f19e1ac/7902801c38c95ad39f5a?experience=power-bi"

# Serve static files (CSS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Function to get a new database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Root@12345",
        database="ajith"
    )

# Function to hash passwords securely
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Function to verify hashed password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# User registration: Stores email and hashed password in MySQL
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    db = get_db_connection()
    cursor = db.cursor()

    # Check if user already exists
    cursor.execute("SELECT * FROM users WHERE email = %s", (username,))
    existing_user = cursor.fetchone()
    if existing_user:
        cursor.close()
        db.close()
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Store the new user with hashed password
    hashed_password = hash_password(password)
    cursor.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (username, hashed_password))
    db.commit()

    cursor.close()
    db.close()
    return {"message": "User registered successfully"}

def authenticate_user(username: str, password: str):
    db = get_db_connection()  # Get a new DB connection
    cursor = db.cursor()
    
    cursor.execute("SELECT password_hash FROM users WHERE email = %s", (username,))
    user = cursor.fetchone()

    cursor.close()
    db.close()  # Close the connection after query

    if user and verify_password(password, user[0]):  # Compare entered password with hashed password
        return True
    return False

# Route for login page
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login1.html", {"request": request})

# Route to process login
@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if authenticate_user(username, password):
        return RedirectResponse(url=POWER_BI_DASHBOARD_URL, status_code=303)
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Route to trigger sentiment analysis pipeline
@app.api_route("/run-script", methods=["GET", "POST"])
async def run_script(request: Request):
    try:
        subprocess.Popen(["python", "sentiment_pipeline_reprocess1.py"])  # Your data collection script
        return {"message": "Pipeline started successfully"}
    except Exception as e:
        return {"error": str(e)}


# Run using: python -m uvicorn api:app --reload


