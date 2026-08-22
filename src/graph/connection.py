import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_driver():
    if not URI or not USER or not PASSWORD:
        raise ValueError("Missing Neo4j credentials — check your .env file against .env.example")
    return GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def test_connection():
    driver = get_driver()
    with driver.session() as session:
        result = session.run("RETURN 'connected' AS status")
        print(result.single()["status"])
    driver.close()


if __name__ == "__main__":
    test_connection()