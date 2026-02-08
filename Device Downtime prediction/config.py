import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# DB Credentials
DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'port': int(os.getenv("DB_PORT")),
    'database': os.getenv("DB_NAME")
}

DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
EMAIL_DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{os.getenv('EMAIL_DB')}"

N8N_DEVICE_WEBHOOK = os.getenv("N8N_DEVICE_WEBHOOK")
print("Webhook URL:", N8N_DEVICE_WEBHOOK)


# Email fetch logic
# def get_plant_email_map():
#     email_map = {}
#     email_engine = create_engine(EMAIL_DATABASE_URL)

#     with email_engine.connect() as conn:
#         query = f"SELECT DISTINCT plant_code, email_to FROM {os.getenv('EMAIL_TABLE')}"
#         result = conn.execute(text(query))
#         for row in result:
#             email_map[row["plant_code"]] = row["email_to"]
    
#     return email_map

def get_plant_email_map():
    mode = os.getenv("ENV_MODE", "TEST")
    test_email = os.getenv("TEST_EMAILS", "sakshi.tandon@amzbizsol.in")
    print("Running in mode:", os.getenv("ENV_MODE"))

    if mode.upper() == "TEST":
        # return dummy test emails for all plants
        return {
            "N202": test_email,
            "N205": test_email,
            "N212": test_email,
            "N225": test_email,
            "N239": test_email,
            "N622": test_email,
            "NE03": test_email,
            "NE25": test_email,
            "NE29": test_email,
            "NE30": test_email,
            "NT45": test_email,
            "NT60": test_email,
            "_default_": test_email
        }

    # else: connect to DB and fetch real emails
    email_map = {}
    email_engine = create_engine(EMAIL_DATABASE_URL)

    with email_engine.connect() as conn:
        query = f"SELECT DISTINCT plant_code, email_to FROM {os.getenv('EMAIL_TABLE')}"
        result = conn.execute(text(query))
        for row in result:
            email_map[row["plant_code"]] = row["email_to"]

    return email_map

