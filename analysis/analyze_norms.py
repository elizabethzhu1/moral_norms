import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Load the data
with open('valid_responses.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract norm similarity ratings and other metrics
df['rating'] = df['norm_similarity'].apply(lambda x: x['rating'])
df['generality'] = df['norm_similarity'].apply(lambda x: x['generality'])
df['logical_relation'] = df['norm_similarity'].apply(lambda x: x['logical_relation'])
df['value'] = df['norm_similarity'].apply(lambda x: x['value'])

# Calculate median rating
median_rating = df['rating'].median()
print(f"Median norm similarity rating: {median_rating}")

# Count ratings in buckets
rating_buckets = {
    'not_similar': len(df[df['rating'].between(1, 3)]),
    'neutral': len(df[df['rating'] == 4]),
    'similar': len(df[df['rating'].between(5, 7)])
}
print("\nRating distribution:")
for bucket, count in rating_buckets.items():
    print(f"{bucket}: {count} responses")

# Count frequency of logical relations and values
logical_relations = Counter(df['logical_relation'])
values = Counter(df['value'])

print("\nLogical relation frequencies:")
for relation, count in logical_relations.items():
    print(f"{relation}: {count}")

print("\nValue frequencies:")
for value, count in values.items():
    print(f"{value}: {count}")

# Create plots
plt.figure(figsize=(15, 10))

# Plot 1: Rating distribution
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='rating', bins=7)
plt.title('Distribution of Norm Similarity Ratings')
plt.xlabel('Rating (1-7)')
plt.ylabel('Count')

# Plot 2: Generality distribution
plt.subplot(2, 2, 2)
sns.histplot(data=df, x='generality', bins=7)
plt.title('Distribution of Generality Ratings')
plt.xlabel('Generality (1-7)')
plt.ylabel('Count')

# Plot 3: Logical relations
plt.subplot(2, 2, 3)
logical_df = pd.DataFrame.from_dict(logical_relations, orient='index', columns=['count'])
logical_df.plot(kind='bar', ax=plt.gca())
plt.title('Frequency of Logical Relations')
plt.xlabel('Logical Relation')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Plot 4: Values
plt.subplot(2, 2, 4)

# Group values into clusters
value_clusters = {
    'Care': ['care', 'harm', 'help', 'protect'],
    'Fairness': ['fairness', 'justice', 'reciprocity', 'equality'],
    'Loyalty': ['loyalty', 'betrayal', 'group', 'community'],
    'Authority': ['authority', 'respect', 'tradition', 'order'],
    'Sanctity': ['sanctity', 'purity', 'disgust', 'cleanliness', 'health'],
    'Honesty': ['honesty', 'deception', 'truth', 'integrity']
}

# Aggregate counts by cluster
cluster_counts = {}
for cluster, values_list in value_clusters.items():
    cluster_counts[cluster] = sum(values.get(v, 0) for v in values_list)

# Create clustered bar plot
cluster_df = pd.DataFrame.from_dict(cluster_counts, orient='index', columns=['count'])
cluster_df.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.title('Frequency of Value Clusters')
plt.xlabel('Value Cluster')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()

# Save all plots to a single figure
plt.savefig('figures/norm_analysis.png')
plt.close()

# Create scatter plot of rating vs generality
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='rating', y='generality', hue='value', alpha=0.6)
plt.title('Relationship between Norm Similarity Rating and Generality')
plt.xlabel('Norm Similarity Rating')
plt.ylabel('Generality Rating')
plt.savefig('figures/rating_vs_generality.png')
plt.close()
