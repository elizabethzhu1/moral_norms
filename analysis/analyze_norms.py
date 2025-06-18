import json
import pandas as pd
from collections import Counter
import re
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

def load_data(file_path: str) -> List[Dict]:
    """Load the JSON data file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_norms(data: List[Dict]) -> List[str]:
    """Extract all norms from the data."""
    norms = []
    for item in data:
        if 'norm' in item and item['norm'] is not None:
            norms.append(item['norm'])
    return norms

def analyze_norm_polarity(norms: List[str]) -> Tuple[List[float], List[str]]:
    """Analyze the sentiment polarity of norms."""
    polarities = []
    categorized_norms = []
    
    for norm in norms:
        try:
            # Get sentiment polarity (-1 to 1)
            polarity = TextBlob(str(norm)).sentiment.polarity
            polarities.append(polarity)
            
            # Categorize norm
            if polarity > 0.1:
                categorized_norms.append('positive')
            elif polarity < -0.1:
                categorized_norms.append('negative')
            else:
                categorized_norms.append('neutral')
        except Exception as e:
            print(f"Warning: Could not analyze norm: {norm}")
            print(f"Error: {str(e)}")
            # Add neutral as default for failed analysis
            polarities.append(0.0)
            categorized_norms.append('neutral')
    
    return polarities, categorized_norms

def identify_rights_and_responsibilities(norms: List[str]) -> Tuple[List[str], List[str]]:
    """Identify norms related to individual rights and social responsibilities."""
    rights_keywords = [
        'right', 'freedom', 'autonomy', 'privacy', 'choice', 'consent',
        'individual', 'personal', 'property', 'liberty'
    ]
    
    responsibility_keywords = [
        'responsibility', 'duty', 'obligation', 'community', 'society',
        'collective', 'group', 'others', 'social', 'public'
    ]
    
    rights_norms = []
    responsibility_norms = []
    
    for norm in norms:
        norm_lower = norm.lower()
        has_rights = any(keyword in norm_lower for keyword in rights_keywords)
        has_responsibilities = any(keyword in norm_lower for keyword in responsibility_keywords)
        
        if has_rights and has_responsibilities:
            rights_norms.append(norm)
            responsibility_norms.append(norm)
        elif has_rights:
            rights_norms.append(norm)
        elif has_responsibilities:
            responsibility_norms.append(norm)
    
    return rights_norms, responsibility_norms

def analyze_norm_categories(norms: List[str]) -> Dict[str, List[str]]:
    """Categorize norms into different types."""
    categories = {
        'personal_ethics': [],
        'social_obligations': [],
        'legal_rights': [],
        'moral_duties': [],
        'cultural_norms': []
    }
    
    for norm in norms:
        norm_lower = norm.lower()
        
        # Personal ethics (individual behavior and character)
        if any(word in norm_lower for word in ['honesty', 'integrity', 'character', 'virtue', 'moral']):
            categories['personal_ethics'].append(norm)
        
        # Social obligations (duties to others)
        if any(word in norm_lower for word in ['help', 'support', 'care', 'respect', 'consider']):
            categories['social_obligations'].append(norm)
        
        # Legal rights (legal protections and entitlements)
        if any(word in norm_lower for word in ['right', 'law', 'legal', 'entitlement']):
            categories['legal_rights'].append(norm)
        
        # Moral duties (ethical obligations)
        if any(word in norm_lower for word in ['duty', 'obligation', 'should', 'must', 'ought']):
            categories['moral_duties'].append(norm)
        
        # Cultural norms (social customs and traditions)
        if any(word in norm_lower for word in ['tradition', 'custom', 'culture', 'society', 'community']):
            categories['cultural_norms'].append(norm)
    
    return categories

def main():
    # Load data
    data = load_data('all_metrics.json')
    norms = extract_norms(data)
    
    print(f"Total number of norms analyzed: {len(norms)}")
    
    # Analyze norm polarity
    polarities, categorized_norms = analyze_norm_polarity(norms)
    polarity_counts = Counter(categorized_norms)
    
    print("\nNorm Polarity Analysis:")
    print(f"Positive norms: {polarity_counts['positive']}")
    print(f"Negative norms: {polarity_counts['negative']}")
    print(f"Neutral norms: {polarity_counts['neutral']}")
    
    # Analyze rights and responsibilities
    rights_norms, responsibility_norms = identify_rights_and_responsibilities(norms)
    
    print("\nRights and Responsibilities Analysis:")
    print(f"Norms mentioning individual rights: {len(rights_norms)}")
    print(f"Norms mentioning social responsibilities: {len(responsibility_norms)}")
    print(f"Norms mentioning both: {len(set(rights_norms) & set(responsibility_norms))}")
    
    # Analyze norm categories
    categories = analyze_norm_categories(norms)
    
    print("\nNorm Categories Analysis:")
    for category, category_norms in categories.items():
        print(f"{category}: {len(category_norms)} norms")
    
    # Print some examples
    print("\nExample norms from each category:")
    for category, category_norms in categories.items():
        if category_norms:
            print(f"\n{category}:")
            for norm in category_norms[:3]:  # Show first 3 examples
                print(f"- {norm}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Polarity distribution
    plt.subplot(2, 2, 1)
    sns.countplot(x=categorized_norms)
    plt.title('Distribution of Norm Polarity')
    plt.xlabel('Polarity')
    plt.ylabel('Count')
    
    # Category distribution
    plt.subplot(2, 2, 2)
    category_counts = {k: len(v) for k, v in categories.items()}
    sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
    plt.title('Distribution of Norm Categories')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # Rights vs Responsibilities
    plt.subplot(2, 2, 3)
    rights_resp_data = {
        'Category': ['Individual Rights', 'Social Responsibilities', 'Both'],
        'Count': [len(rights_norms), len(responsibility_norms), 
                 len(set(rights_norms) & set(responsibility_norms))]
    }
    sns.barplot(data=pd.DataFrame(rights_resp_data), x='Category', y='Count')
    plt.title('Rights vs Responsibilities')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('analysis/norm_analysis.png')
    plt.close()

if __name__ == "__main__":
    main() 