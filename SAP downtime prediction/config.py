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

DATABASE_URL = (
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

EMAIL_DATABASE_URL = (
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{os.getenv('EMAIL_DB')}"
)

N8N_EMAIL_WEBHOOK = os.getenv("N8N_EMAIL_WEBHOOK")
DEFAULT_EMAIL = os.getenv("DEFAULT_EMAIL")

ENV_MODE = os.getenv("ENV_MODE", "TEST").upper()
TEST_EMAIL = os.getenv("TEST_EMAILS", DEFAULT_EMAIL)

# -------------------
# Email fetch logic
# -------------------
def get_plant_email_map():
    if ENV_MODE == "TEST":
        # Return dummy mapping for testing
        return {
            "N202": TEST_EMAIL,
            "N205": TEST_EMAIL,
            "N212": TEST_EMAIL,
            "N225": TEST_EMAIL,
            "N239": TEST_EMAIL,
            "N622": TEST_EMAIL,
            "NE03": TEST_EMAIL,
            "NE25": TEST_EMAIL,
            "NE29": TEST_EMAIL,
            "NE30": TEST_EMAIL,
            "NT45": TEST_EMAIL,
            "NT60": TEST_EMAIL,
            "_default_": TEST_EMAIL
        }

    # In PROD mode, fetch from DB
    email_map = {}
    email_engine = create_engine(EMAIL_DATABASE_URL)
    with email_engine.connect() as conn:
        query = f"SELECT DISTINCT plant_code, email_to FROM {os.getenv('EMAIL_TABLE')}"
        result = conn.execute(text(query))
        for row in result:
            email_map[row["plant_code"]] = row["email_to"]

    if "_default_" not in email_map:
        email_map["_default_"] = DEFAULT_EMAIL
    return email_map
