"""
Semantic Router: Advanced Exploration Example

This script demonstrates an expanded implementation of the Semantic Router library
for semantic-based query routing with visualization, evaluation, and integration examples.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv

# Semantic Router imports
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import OpenAIEncoder

# Load environment variables from .env file
load_dotenv()
print("Semantic Router Advanced Example")

# Get the OpenAI API key from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")

# ----- Helper Functions -----

def evaluate_router(router, test_queries: List[Dict], verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate router performance on a set of test queries.
    
    Args:
        router: Initialized SemanticRouter
        test_queries: List of dictionaries with 'query' and 'expected_route' keys
        verbose: Whether to print detailed results
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for item in test_queries:
        query = item['query']
        expected = item['expected_route']
        
        start_time = time.time()
        result = router(query)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        correct = result.name == expected
        
        results.append({
            'query': query,
            'expected_route': expected,
            'predicted_route': result.name,
            'correct': correct,
            'processing_time_ms': processing_time
        })
        
        if verbose:
            status = "✓" if correct else "✗"
            print(f"{status} Query: '{query}' → Predicted: {result.name}, Expected: {expected}")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Print summary statistics
    if verbose:
        accuracy = df['correct'].mean() * 100
        avg_time = df['processing_time_ms'].mean()
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average processing time: {avg_time:.2f} ms")
        
        # Per-route statistics
        print("\nPer-route accuracy:")
        route_accuracy = df.groupby('expected_route')['correct'].mean() * 100
        for route, acc in route_accuracy.items():
            print(f"  {route}: {acc:.2f}%")
    
    return df

def visualize_query_distribution(eval_df: pd.DataFrame) -> None:
    """
    Visualize the distribution and accuracy of query routing.
    """
    # Create a confusion matrix
    routes = sorted(list(set(eval_df['expected_route'].unique()) | set(eval_df['predicted_route'].unique())))
    confusion = pd.crosstab(
        eval_df['expected_route'], 
        eval_df['predicted_route'],
        rownames=['Expected'],
        colnames=['Predicted']
    )
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, interpolation='nearest', cmap='Blues')
    plt.title('Routing Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(routes))
    plt.xticks(tick_marks, routes, rotation=45)
    plt.yticks(tick_marks, routes)
    
    # Add values to cells
    for i in range(len(routes)):
        for j in range(len(routes)):
            if i < confusion.shape[0] and j < confusion.shape[1]:
                plt.text(j, i, format(confusion.iloc[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if confusion.iloc[i, j] > 2 else "black")
    
    plt.tight_layout()
    plt.ylabel('Expected Route')
    plt.xlabel('Predicted Route')
    
    # Save the figure
    plt.savefig('routing_confusion_matrix.png')
    print("Confusion matrix saved as 'routing_confusion_matrix.png'")
    
    # Additional visualization: query processing time
    plt.figure(figsize=(12, 6))
    eval_df.boxplot(column='processing_time_ms', by='expected_route')
    plt.title('Query Processing Time by Route Type')
    plt.suptitle('')  # Remove pandas default title
    plt.ylabel('Processing Time (ms)')
    plt.xlabel('Route Type')
    plt.tight_layout()
    plt.savefig('processing_time_by_route.png')
    print("Processing time visualization saved as 'processing_time_by_route.png'")

class QueryRouter:
    """
    A class that wraps the SemanticRouter with additional functionality
    including handlers for each route type and confidence thresholds.
    """
    
    def __init__(self, encoder, routes: List[Route], confidence_threshold: float = 0.5):
        """
        Initialize the QueryRouter.
        
        Args:
            encoder: The encoder to use for semantic embeddings
            routes: List of Route objects
            confidence_threshold: Minimum confidence score to accept a route match
        """
        self.router = SemanticRouter(encoder=encoder)
        self.routes = routes
        self.confidence_threshold = confidence_threshold
        self.handlers = {}
        
        # Add routes to the router
        self.router.add(routes)
        print(f"Initialized router with {len(routes)} routes")
        time.sleep(2)  # Give time for index to initialize
        
    def add_handler(self, route_name: str, handler_func: callable) -> None:
        """
        Add a handler function for a specific route.
        
        Args:
            route_name: Name of the route
            handler_func: Function to handle queries for this route
        """
        self.handlers[route_name] = handler_func
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the router and appropriate handler.
        
        Args:
            query: The user's query text
            
        Returns:
            Dictionary with routing result and response
        """
        # Get routing result
        result = self.router(query)
        route_name = result.name
        
        # Check if we have a handler for this route
        response = None
        if route_name in self.handlers:
            response = self.handlers[route_name](query)
        else:
            response = f"No handler available for route: {route_name}"
            
        return {
            "query": query,
            "route": route_name,
            "response": response
        }
    
    def save_routes_to_file(self, filepath: str) -> None:
        """Save route definitions to a JSON file for later reuse."""
        routes_data = []
        for route in self.routes:
            route_dict = {
                "name": route.name,
                "utterances": route.utterances
            }
            routes_data.append(route_dict)
            
        with open(filepath, 'w') as f:
            json.dump(routes_data, f, indent=2)
        
        print(f"Saved {len(routes_data)} routes to {filepath}")
    
    @classmethod
    def load_routes_from_file(cls, filepath: str) -> List[Route]:
        """Load route definitions from a JSON file."""
        with open(filepath, 'r') as f:
            routes_data = json.load(f)
            
        routes = []
        for route_dict in routes_data:
            route = Route(
                name=route_dict["name"],
                utterances=route_dict["utterances"]
            )
            routes.append(route)
            
        print(f"Loaded {len(routes)} routes from {filepath}")
        return routes

# ----- Main Application Logic -----

def main():
    # Initialize the encoder
    encoder = OpenAIEncoder(openai_api_key=openai_api_key)
    
    # Define a more comprehensive set of routes
    routes = [
        Route(
            name="weather",
            utterances=[
                "What's the weather like today?",
                "Will it rain tomorrow?",
                "Is it going to be sunny this weekend?",
                "What's the temperature outside?",
                "Do I need an umbrella today?",
                "How hot will it be tomorrow?",
                "Is there a storm coming?",
                "What's the forecast for next week?",
                "How's the weather in New York?",
                "Will it snow tonight?"
            ]
        ),
        Route(
            name="calculator",
            utterances=[
                "What's 5 plus 3?",
                "Calculate 10 multiplied by 4",
                "What's the square root of 16?",
                "Solve 3x + 5 = 20",
                "Convert 10 kilometers to miles",
                "What's 20% of 50?",
                "Calculate the area of a circle with radius 5",
                "What's the derivative of x²?",
                "Add 3/4 and 1/2",
                "Calculate the compound interest on $1000 at 5% for 3 years"
            ]
        ),
        Route(
            name="general_knowledge",
            utterances=[
                "Who was the first president of the United States?",
                "What is the capital of France?",
                "When was the Declaration of Independence signed?",
                "What's the tallest mountain in the world?",
                "Who wrote Romeo and Juliet?",
                "What's the chemical symbol for gold?",
                "How many planets are in our solar system?",
                "Who painted the Mona Lisa?",
                "What's the largest ocean on Earth?",
                "What year did World War II end?"
            ]
        ),
        Route(
            name="recommendations",
            utterances=[
                "Can you recommend a good book to read?",
                "What movie should I watch tonight?",
                "What's a good restaurant in Chicago?",
                "Recommend a podcast about history",
                "What's a good gift for my mom's birthday?",
                "Can you suggest some exercises for beginners?",
                "What's a good place to visit in Europe?",
                "Recommend a healthy breakfast recipe",
                "What video game would you recommend?",
                "Can you suggest a good online course for machine learning?"
            ]
        ),
        Route(
            name="personal_assistant",
            utterances=[
                "Remind me to call mom at 5pm",
                "Set an alarm for 7am tomorrow",
                "What's on my calendar for today?",
                "Add milk to my shopping list",
                "Schedule a meeting with John for Thursday",
                "What appointments do I have next week?",
                "Send a message to Sarah",
                "Create a to-do list for my project",
                "What's the status of my Amazon order?",
                "How much time do I have until my next meeting?"
            ]
        ),
        Route(
            name="technical_support",
            utterances=[
                "My computer won't turn on",
                "How do I reset my password?",
                "Why is my internet connection so slow?",
                "How do I update my operating system?",
                "My printer is showing an error",
                "How do I connect my phone to WiFi?",
                "The app keeps crashing on my phone",
                "How do I transfer files between devices?",
                "My screen is frozen and won't respond",
                "How do I backup my data?"
            ]
        )
    ]
    
    # Initialize our query router
    query_router = QueryRouter(encoder, routes)
    
    # Add handler functions for each route
    query_router.add_handler("weather", lambda q: f"Weather processing: {q}")
    query_router.add_handler("calculator", lambda q: f"Calculation result: {q}")
    query_router.add_handler("general_knowledge", lambda q: f"Knowledge lookup: {q}")
    query_router.add_handler("recommendations", lambda q: f"Recommendation for: {q}")
    query_router.add_handler("personal_assistant", lambda q: f"Assistant action: {q}")
    query_router.add_handler("technical_support", lambda q: f"Technical support: {q}")
    
    # Save route definitions for future use
    query_router.save_routes_to_file("saved_routes.json")
    
    # Define test queries for evaluation
    test_queries = [
        {"query": "What's the temperature in Miami today?", "expected_route": "weather"},
        {"query": "Will it be foggy in San Francisco tomorrow?", "expected_route": "weather"},
        {"query": "Calculate 25 divided by 5", "expected_route": "calculator"},
        {"query": "What's the integral of sin(x)?", "expected_route": "calculator"},
        {"query": "Who discovered penicillin?", "expected_route": "general_knowledge"},
        {"query": "When was the lightbulb invented?", "expected_route": "general_knowledge"},
        {"query": "Can you suggest a science fiction book?", "expected_route": "recommendations"},
        {"query": "What's a good Italian restaurant near me?", "expected_route": "recommendations"},
        {"query": "Set a timer for 10 minutes", "expected_route": "personal_assistant"},
        {"query": "What meetings do I have tomorrow?", "expected_route": "personal_assistant"},
        {"query": "My laptop battery isn't charging", "expected_route": "technical_support"},
        {"query": "How do I fix a blue screen error?", "expected_route": "technical_support"},
        # Edge cases and potential confusions
        {"query": "What's the formula for calculating compound interest?", "expected_route": "calculator"},
        {"query": "Who invented the calculator?", "expected_route": "general_knowledge"},
        {"query": "Can you recommend a weather app?", "expected_route": "recommendations"},
        {"query": "What's the history of weather forecasting?", "expected_route": "general_knowledge"},
        {"query": "Remind me to check the weather tomorrow", "expected_route": "personal_assistant"},
        {"query": "My weather app is not working", "expected_route": "technical_support"}
    ]
    
    # Evaluate the router on test queries
    print("\n--- Evaluation Results ---")
    eval_df = evaluate_router(query_router.router, test_queries)
    
    # Visualize evaluation results
    try:
        visualize_query_distribution(eval_df)
    except Exception as e:
        print(f"Visualization error: {str(e)}")
    
    # Interactive testing loop
    print("\n--- Interactive Testing ---")
    print("Type a query to test routing (or 'exit' to quit):")
    
    while True:
        user_input = input("\nQuery: ")
        if user_input.lower() in ('exit', 'quit'):
            break
            
        try:
            result = query_router.process_query(user_input)
            print(f"Routed to: {result['route']}")
            print(f"Response: {result['response']}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    
    print("Demo completed.")

if __name__ == "__main__":
    main()