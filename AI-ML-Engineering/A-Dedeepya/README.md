 YouTube Video Recommendation System

A sophisticated machine learning-powered recommendation engine that combines YouTube data scraping with a two-tower neural network architecture to provide personalized video recommendations.

 🎯 Project Overview

This project implements a complete recommendation pipeline consisting of:
- **YouTube Data Scraping**: Automated collection of video metadata and transcripts
- **Two-Tower Neural Network**: Deep learning model for personalized recommendations
- **Interactive Web Interface**: Streamlit-based UI for real-time recommendations
- **FAISS Integration**: High-performance similarity search for scalable recommendations

🏗️ Architecture

 Core Components

1. Data Collection Layer (youtube_scraper/)
   - YouTube API integration for video metadata
   - Transcript extraction using YouTube Transcript API
   - Airtable integration for data storage and deduplication

2. Machine Learning Layer** (model.py , util.py)
   - Two-Tower neural network architecture
   - User and item embeddings with contextual features
   - FAISS-powered approximate nearest neighbor search

3. Application Layer** (app.py, main.py)
   - Streamlit web interface
   - Real-time recommendation generation
   - Interactive user profiling

 Quick Start

 Prerequisites

```bash
# Core dependencies
pip install torch streamlit faiss-cpu pandas numpy scikit-learn

# YouTube scraping dependencies
pip install requests youtube-transcript-api google-api-python-client
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ZovxLabs-Internship-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r youtube_scraper/requirements.txt
   pip install torch streamlit faiss-cpu pandas numpy scikit-learn google-api-python-client
   ```

3. **Configure API Keys**
   ```bash
   # Set your YouTube Data API key in:
   # - main.py (line 11)
   # - youtube_scraper/ytscrap.py (line 6)
   
   # Set Airtable credentials in youtube_scraper/airtablescript.py (lines 8-10)
   ```

### Running the Application

#### Option 1: Simple Recommendation Interface
```bash
streamlit run main.py
```
Access the application at `http://localhost:8501`

#### Option 2: Advanced Two-Tower Model Interface
```bash
# First, ensure you have a trained model (model.pth)
streamlit run app.py
```

#### Option 3: Data Collection Pipeline
```bash
cd youtube_scraper
python ytscrap.py          # Scrape videos and transcripts
python airtablescript.py   # Upload to Airtable
```

## 🧠 Model Architecture

### Two-Tower Neural Network

The recommendation system uses a two-tower architecture that learns separate representations for users and items:

```python
User Tower:
├── User Embedding (32D)
├── Time-of-Day Embedding (32D)
├── Device Type Embedding (32D)
└── Fully Connected Layer (96D → 32D)

Item Tower:
├── Item Embedding (32D)
├── Category Embedding (32D)
└── Fully Connected Layer (64D → 32D)

Final Score: dot_product(user_vector, item_vector)
```

### Key Features

- **Contextual Embeddings**: Time-of-day and device type awareness
- **Scalable Architecture**: FAISS integration for efficient similarity search
- **Real-time Inference**: Optimized for low-latency recommendations
- **Flexible Item Representation**: Category-aware item embeddings

## 📁 Project Structure

```
ZovxLabs-Internship-main/
├── app.py                    # Advanced Streamlit interface (Two-Tower)
├── main.py                   # Simple Streamlit interface (YouTube API)
├── model.py                  # Two-Tower model definition
├── util.py                   # FAISS utilities and recommendation logic
├── model.pth                 # Trained model weights (generated)
└── youtube_scraper/
    ├── ytscrap.py           # YouTube video and transcript scraper
    ├── airtablescript.py    # Airtable integration script
    ├── dag.py               # (Empty) Airflow DAG placeholder
    ├── requirements.txt     # Scraping dependencies
    └── *.jsonl             # Generated transcript data files
```

## 🔧 Technical Details

### Data Flow

1. **Collection**: YouTube videos → Transcripts → Metadata
2. **Processing**: Raw data → Embeddings → Feature vectors
3. **Training**: User-item interactions → Neural network weights
4. **Inference**: User profile → Similarity search → Ranked recommendations

### Model Training

The system supports two training approaches:

1. **Simple Training** (`main.py`): 
   - Real-time training on YouTube trending videos
   - Simulated user interactions
   - BCELoss optimization

2. **Advanced Training** (`model.py`):
   - Pre-trained embeddings with categorical features
   - Batch training with custom datasets
   - Dot-product similarity optimization

### Performance Considerations

- **FAISS Indexing**: O(log n) similarity search
- **Embedding Normalization**: L2 normalization for stable similarities
- **Batch Processing**: Efficient GPU/CPU utilization
- **Memory Optimization**: Compressed embedding storage

## 🎮 Usage Examples

### Basic Recommendation

```python
# Generate recommendations for a user
user_vector = get_user_vector(model, user_id=42, 
                             time_str="Evening", 
                             device_str="Mobile")
recommendations = recommend_top_k(user_vector, item_vectors, 
                                 item_ids, top_k=10)
```

### Data Collection

```python
# Scrape channel videos
videos = fetch_all_videos_with_titles(API_KEY, CHANNEL_ID)

# Extract transcripts
for video in videos:
    transcript = fetch_transcript(video['video_id'])
    # Process transcript...
```

## 🔐 Security & Configuration

### API Keys Required

- **YouTube Data API v3**: For video metadata and trending content
- **Airtable API** (Optional): For data storage and management

### Environment Setup

Create a `.env` file (not included in repo):
```bash
YOUTUBE_API_KEY=your_youtube_api_key
AIRTABLE_API_KEY=your_airtable_key
AIRTABLE_BASE_ID=your_base_id
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Future Enhancements

- [ ] Multi-modal embeddings (text + visual features)
- [ ] Online learning with user feedback
- [ ] A/B testing framework
- [ ] Production deployment with Docker
- [ ] Real-time data pipeline with Apache Kafka
- [ ] Advanced evaluation metrics (NDCG, MAP)

## 🐛 Known Issues

- Model requires pre-training before using `app.py`
- API rate limits may affect large-scale scraping
- FAISS index needs rebuilding when model changes

## 📄 License

This project is part of ZovxLabs internship program. Please contact the organization for licensing details.

## 👥 Team

**ZovxLabs Internship Project**  
*Machine Learning & Data Engineering Team*

---

**Tech Stack**: PyTorch • Streamlit • FAISS • YouTube API • Scikit-learn • Pandas • NumPy
