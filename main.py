"""
Interactive sentiment analysis demo
"""

from bayes import BayesClassifier

def main():
    print("=== Naive Bayes Sentiment Classifier ===\n")
    
    classifier = BayesClassifier()
    classifier.setup('train_test2')  # Change to your training folder
    
    print("\nEnter reviews to analyze (or 'quit' to exit)\n")
    
    while True:
        text = input("Review: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not text:
            continue
            
        result = classifier.classify(text)
        emoji = "😊" if result == "positive" else "😞"
        print(f"→ {result.upper()} {emoji}\n")

if __name__ == "__main__":
    main()
