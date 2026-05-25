# Deep Learning Strategy for ig-autobot

This document outlines the strategy for evolving the `ig-autobot` from a rule-based automation tool into a data-driven predictive system using Deep Learning.

## Phase 8: Predictive Modeling & Optimization

### Step 1: Data Collection and Preparation

The most crucial step is collecting high-quality data. Every post and its performance must be logged systematically.

**Data Points to Collect:**
* **Image Data:** The original media file.
* **Caption Data:** The generated text.
* **Hashtags:** The specific cluster used.
* **Timestamps:** GST and UTC post times.
* **Engagement Metrics:** Reach, Impressions, Shares (Primary), Saves, Likes, Comments.
* **CTA Performance:** Website clicks via tracking parameters (UTM).
* **Book Sales Data:** Conversion data attributed via custom links.

```python
import pandas as pd
import numpy as np

# Sample Data Structure
data = {
    'post_id': [1, 2, 3, 4, 5],
    'caption': ['...', '...', '...', '...', '...'],
    'likes': [150, 200, 120, 300, 80],
    'comments': [10, 25, 5, 40, 3],
    'book_sales': [1, 3, 0, 8, 0] # Ultimate target
}
df_posts = pd.DataFrame(data)
df_posts['engagement_score'] = df_posts['likes'] + (df_posts['comments'] * 2)
```

### Step 2: Feature Engineering

Transforming raw media into numerical features that a neural network can process.

#### A. Image Features
Use pre-trained Convolutional Neural Networks (CNN) like **ResNet** or **EfficientNet** to extract embeddings that capture the visual "vibe" of successful posts.

#### B. Text Features
Use NLP models (e.g., **DistilBERT**) to represent captions as high-dimensional vectors.
* **Sentiment Analysis:** Does "cynical wit" outperform "earnest philosophy"?
* **Readability:** Do shorter, simpler hooks drive more shares?

#### C. Temporal Features
* Hour of day, day of week, and alignment with UAE peak times.

### Step 3: Model Development

Training a multi-output regression model to predict both **Engagement** and **Sales**.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Multi-output model architecture
input_layer = Input(shape=(feature_vector_size,))
hidden = Dense(256, activation='relu')(input_layer)
hidden = Dense(128, activation='relu')(hidden)

sales_output = Dense(1, activation='linear', name='book_sales_output')(hidden)
engagement_output = Dense(1, activation='linear', name='engagement_output')(hidden)

model = Model(inputs=input_layer, outputs=[sales_output, engagement_output])
```

### Step 4: Content Optimization (The "Selection" Engine)

Once trained, the bot no longer picks a post at random. It:
1. Generates 5 candidate captions/images.
2. Extracts features for all 5.
3. Uses `model.predict()` to score them.
4. **Selects the candidate with the highest predicted ROI.**

### Step 5: Integration and Feedback Loop

* **Continuous Learning:** Resulting metrics from every post are fed back into the training set.
* **Retraining:** The model is periodically retrained to adapt to shifting algorithm trends (like the mid-2026 shift toward Shares).

---

## Technical Assessment

### Strengths
* **Data-Driven ROI:** Moves beyond "guessing" what works to mathematical prediction of sales.
* **Persona Refinement:** Can objectively identify which "sub-voices" of the Professional Failure Expert persona resonate most.
* **Algorithmic Adaptation:** Automatically learns when Instagram shifts priorities (e.g., from Saves to Shares).

### Challenges
* **Data Volume:** Deep learning requires hundreds, if not thousands, of data points to become accurate.
* **Sales Attribution:** Requires tight integration with the storefront (e.g., UTM tracking) to provide a clean "target" variable.
* **Compute:** Extracting embeddings (BERT/ResNet) requires GPU resources during the generation phase.
