import os
import sys

# Configuration class for environment variables and settings
class Config:
    LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_URL = os.getenv('QDRANT_URL')
    OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    DATA_PATHS = {
        'sustainability': os.path.abspath("data/Dataset 1 (Sustainability Research Results).xlsx"),
        'christmas': os.path.abspath("/data/Dataset_2_Christmas_Research_Results.xlsx")
    }
    INSTRUCTIONS = {
        'sustainability': """You are parsing a file containing an NxN breakdown of the results to a survey commissioned by Bounce 
                             Insights asking consumers in the UK various questions around how important is sustainability to 
                             consumers when they are buying products in general & how engaged are consumers with sustainable brands or products""",
        'christmas': """You are parsing a file containing an NxN breakdown of the results to a survey commissioned by Bounce Insights 
                        asking consumers in Ireland various questions to understand the consumers' plans for Christmas, what their 
                        plans are overall and with spending."""
    }
    COLLECTION_NAME = "bounce_insights_project"