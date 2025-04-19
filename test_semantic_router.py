# test_semantic_router.py
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

print("Successfully imported Semantic Router!")

# Get the OpenAI API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
else:
    try:
        # Initialize the encoder with the correct parameter name
        encoder = OpenAIEncoder(openai_api_key=openai_api_key)
        
        # Define some routes
        weather_route = Route(
            name="weather", 
            utterances=["What's the weather like?", "Will it rain today?", "Is it going to be sunny?"]
        )

        calculator_route = Route(
            name="calculator",
            utterances=["Calculate 5+3", "What's 10 multiplied by 4?", "Solve this equation"]
        )

        # Create the router
        router = SemanticRouter(encoder=encoder)
        
        # Add routes explicitly
        print("Adding routes to the router...")
        router.add([weather_route, calculator_route])
        
        # Wait a moment to ensure the index is ready
        print("Waiting for index to be ready...")
        time.sleep(3)
        
        # Test the router
        test_query = "What's the temperature in New York?"
        print(f"Routing query: '{test_query}'")
        result = router(test_query)
        print(f"Query was routed to: {result.name}")
        
    except Exception as e:
        print(f"Error when testing router: {e}")
        import traceback
        traceback.print_exc()

print("Setup complete!")