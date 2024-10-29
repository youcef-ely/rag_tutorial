import os
import torch
import logging
import pretty_errors


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GENERATOR_MODEL_NAME = "tiiuae/falcon-7b-instruct"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    LANGCHAIN_ENDPOINT = 'https://api.smith.langchain.com'
    FASTAPI_HOST = "127.0.0.1"
    FASTAPI_PORT = 8000  
    COLLECTION_NAME = "dysgraphia_paper" 

    
    @classmethod
    def validate_env_variables(cls):
        def check_api_key(key_name):
            key_value = os.getenv(key_name)
            if key_value is None:
                logging.error(f'{key_name} is not set!')
                raise ValueError(f'{key_name} cannot be None')
            return key_value

        try:
            # Environment variable validation
            cls.LANGCHAIN_API_KEY = check_api_key('LANGCHAIN_API_KEY')
            cls.HUGGINGFACEHUB_API_TOKEN = check_api_key('HUGGINGFACEHUB_API_TOKEN')

            # Additional environment configuration (optional)
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
            os.environ['LANGCHAIN_ENDPOINT'] = cls.LANGCHAIN_ENDPOINT

            

        except ValueError as e:
            logging.critical(f'Critical error: {e}')
            raise
        
        logging.info('API Keys and configuration validated successfully!')

# Call validate_env_variables once to ensure environment is set up
Config.validate_env_variables()
