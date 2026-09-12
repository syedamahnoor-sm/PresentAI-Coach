# 🎤 PresentAI Coach

### Multimodal Agentic AI System for Presentation Analysis and Personalized Coaching

PresentAI Coach is an AI-powered presentation coaching system that analyzes a presenter's **visual delivery, speech, and spoken content** and converts those observations into personalized, actionable coaching.

The system combines **Computer Vision, Machine Learning, Speech Recognition, and Agentic AI** to evaluate presentation performance across four core dimensions:

- 🧍 Posture
- 👋 Gestures
- 👁️ Eye Contact
- 🎙️ Speech Delivery

Unlike a simple scoring application, PresentAI Coach uses a **LangGraph-based multi-step AI coaching workflow** powered by **Google Gemini**. The agent reasons over objective analytics, identifies the most important improvement areas, generates personalized coaching, critiques its own response, and produces a refined final coaching plan.

The application supports both:

- 🎥 **Real-Time Presentation Practice**
- ☁️ **Uploaded Presentation Video Analysis**

---

## ✨ Key Features

### 🎥 Real-Time Presentation Analysis

PresentAI Coach can analyze a presentation directly from the user's browser using camera and microphone access through WebRTC.

During a session, the system tracks:

- Posture and body alignment
- Gesture activity
- Eye-contact/gaze behavior
- Speech pace
- Filler words
- Pauses
- Speech fluency

The visual pipeline is optimized to avoid running expensive computer-vision models unnecessarily on every frame.

Browser microphone audio is captured through WebRTC and passed to the shared speech-analysis pipeline, allowing live presentations to receive the same speech evaluation used for uploaded recordings.

---

### ☁️ Uploaded Video Analysis

Users can upload recorded presentations in common video formats such as:

- MP4
- MOV
- AVI
- MKV

The system extracts and analyzes both the visual and audio components of the presentation before producing a unified performance report.

---

### 🧍 ML-Based Posture Analysis

Posture evaluation uses a custom machine-learning pipeline built on top of **MediaPipe Pose Landmarker** and **Face Landmarker** features.

The posture model uses geometric features including:

- Shoulder alignment
- Shoulder depth difference
- Torso alignment
- Torso depth
- Head position
- Head tilt
- Head-to-shoulder relationships
- Face alignment
- Facial depth relationships

A trained **Random Forest classifier** evaluates these features and contributes to the final posture score.

Temporal smoothing and hysteresis are used to reduce unstable frame-to-frame predictions.

---

### 👋 Gesture Analysis

Gesture analysis evaluates temporal hand and body movement rather than relying only on individual frames.

The system considers factors such as:

- Hand movement
- Gesture-zone activity
- Presenter movement
- Excessive motion
- Frozen/static delivery
- Movement stability

This helps distinguish between useful expressive movement and potentially distracting delivery patterns.

---

### 👁️ Eye Contact Analysis

Face landmarks are used to estimate presentation-oriented gaze behavior.

The system considers:

- Nose alignment
- Facial orientation
- Eye position
- Cheek depth relationships
- Horizontal gaze displacement

This is designed as a **presentation-delivery heuristic**, not medical or clinical eye tracking.

---

### 🎙️ Speech Analysis

PresentAI Coach uses **Faster-Whisper** for speech recognition.

The speech pipeline evaluates:

- Word count
- Words per minute (WPM)
- Filler words
- Filler-word rate
- Pauses
- Long pauses
- Fluency
- Recognition confidence
- Full presentation transcript

The same speech-analysis logic is shared between uploaded-video and live-presentation workflows.

For live sessions, browser microphone audio is captured through WebRTC and processed after the presentation is completed. For uploaded presentations, audio is extracted from the video and passed through the same speech-analysis pipeline.

---

## 🤖 Agentic AI Presentation Coach

One of the core features of PresentAI Coach is its **agentic coaching workflow**.

The system does not ask the LLM to determine posture, eye contact, speech scores, or other objective measurements.

Instead:

1. Computer Vision, ML, and speech-processing modules generate objective evidence.
2. The evidence is passed to the AI coaching workflow.
3. Specialized reasoning stages analyze different aspects of the presentation.
4. The agent determines the highest-impact coaching priorities.
5. Personalized coaching is generated.
6. A critic evaluates the coaching for consistency and unsupported claims.
7. The response is approved or revised before being shown to the user.

### Agent Workflow

```text
Presentation
     │
     ▼
Objective Analytics
     │
     ├── Posture
     ├── Gestures
     ├── Eye Contact
     └── Speech
     │
     ▼
Evidence Grounding
     │
     ▼
Visual Delivery Analysis
     │
     ▼
Speech Delivery Analysis
     │
     ▼
Content Analysis
     │
     ▼
Priority Reasoning
     │
     ▼
Personalized AI Coach
     │
     ▼
Critic / Quality Review
     │
     ├── Approved ──────────┐
     │                      │
     └── Revision Required  │
              │             │
              ▼             │
           Revision         │
              │             │
              └─────────────┘
                    │
                    ▼
             Final Coaching
```

The workflow is orchestrated using **LangGraph**, with **Google Gemini** used for language reasoning and structured coaching generation.

This separation keeps the analytical measurements deterministic while allowing the AI layer to provide contextual and natural coaching.

---

## 🧠 Coaching Output

After analyzing a presentation, the AI coach can generate:

- Overall coaching summary
- Key strengths
- Prioritized improvement areas
- Explanation of why each issue matters
- Specific improvement techniques
- Content and clarity feedback
- A practical plan for the next attempt
- A primary focus for the presenter

The agent is explicitly instructed not to overwrite objective analytical scores or make unsupported medical or ergonomic claims.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   PresentAI Coach   │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        Live Browser Session           Uploaded Video
       Camera + Microphone                    │
                 │                           │
                 └─────────────┬─────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
         VisualAnalyzer               SpeechAnalyzer
                 │                           │
      ┌──────────┼──────────┐                │
      │          │          │                │
      ▼          ▼          ▼                │
   Posture    Gestures     Gaze              │
      │          │          │                │
      └──────────┼──────────┘                │
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                        SessionManager
                               │
                               ▼
                    Unified Analysis Report
                               │
                               ▼
                        LangGraph Agent
                               │
                               ▼
                       Gemini Reasoning
                               │
                               ▼
                Personalized AI Coaching
                               │
                               ▼
                         Streamlit UI
```

---

## 🔄 Shared Analysis Architecture

The application intentionally avoids maintaining separate analytical implementations for live and uploaded presentations.

```text
             VisualAnalyzer
              /          \
         Live Video      Upload
             ↓             ↓
       Browser Mic     Video Audio
              \          /
              SpeechAnalyzer
                    ↓
              SessionManager
                    ↓
              Unified Report
                    ↓
                 AI Coach
```

This allows both modes to use the same underlying analytical logic and keeps results consistent across the application.

---

## 🛠️ Technology Stack

### Programming Language

- Python 3.12

### Computer Vision

- OpenCV
- MediaPipe Tasks
- MediaPipe Pose Landmarker
- MediaPipe Face Landmarker

### Machine Learning

- Scikit-learn
- Random Forest
- NumPy
- Pandas

### Speech Processing

- Faster-Whisper
- FFmpeg
- SoundDevice
- SciPy

### Agentic AI

- LangGraph
- LangChain
- Google Gemini
- LangChain Google GenAI
- Pydantic structured outputs

### User Interface & Real-Time Communication

- Streamlit
- Streamlit WebRTC
- PyAV
- WebRTC
- STUN/TURN connectivity

### Deployment

- Streamlit Community Cloud
- Metered TURN/STUN infrastructure for cloud WebRTC connectivity

---

## 📁 Project Structure

```text
PresentAI Coach/
│
├── app.py
├── ai_coach_agent.py
├── visual_analyzer.py
├── video_analyzer.py
├── live_analyzer.py
├── speech_analyzer.py
├── session_manager.py
├── posture_scorer.py
├── gesture_analyzer.py
├── gaze_analyzer.py
├── feature_extractor.py
├── media_utils.py
├── vision.py
├── dataset_processor.py
├── video_processor.py
├── train_classifier.py
├── evaluate_classifier.py
├── generate_submission_results.ipynb
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── pose_landmarker_lite.task
│   ├── face_landmarker.task
│   ├── posture_classifier.pkl
│   └── posture_classifier_eval.pkl
│
├── results/
│   ├── 01_class_distribution.png
│   ├── 02_dataset_source_distribution.png
│   ├── 03_confusion_matrix.png
│   ├── 04_roc_curve.png
│   ├── 05_model_performance.png
│   ├── model_metrics.csv
│   └── classification_report.csv
│
└── tests/
    ├── __init__.py
    ├── test_posture_scorer.py
    ├── test_speech_analyzer.py
    ├── test_session_manager.py
    └── test_ai_coach.py
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/syedamahnoor-sm/PresentAI-Coach
cd "PresentAI-Coach"
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

FFmpeg must also be installed and available through the system PATH for uploaded-video audio extraction.

---

## 🔑 Environment Configuration

PresentAI Coach uses environment variables for Gemini-powered AI coaching and authenticated TURN connectivity.

For local development, create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
PRESENTAI_LLM_MODEL=gemini-3.6-flash
TURN_USERNAME=your_turn_username
TURN_CREDENTIAL=your_turn_credential
```

The variables are used as follows:

- `GOOGLE_API_KEY` — authenticates requests used by the Gemini-powered coaching workflow.
- `PRESENTAI_LLM_MODEL` — specifies the Gemini model used by the AI coaching agent.
- `TURN_USERNAME` — username used for authenticated TURN connectivity.
- `TURN_CREDENTIAL` — credential used for authenticated TURN connectivity.

For **Streamlit Community Cloud**, secret values should be configured through the application's Streamlit Secrets rather than committed to the repository.

> **Important:** Never commit `.env`, API keys, TURN credentials, or other secrets to Git.

Ensure `.gitignore` contains:

```text
.env
venv/
__pycache__/
*.pyc
```

---

## ▶️ Running the Application

Start the Streamlit application:

```powershell
streamlit run app.py
```

The application will open in the browser.

From the home page, select either:

### 🎥 Real-Time Practice

Start a browser-based presentation session using camera and microphone access.

The live workflow analyzes:

- Posture
- Gestures
- Eye contact
- Speech pace
- Filler words
- Pauses
- Fluency

When the presentation is finished, the visual and speech results are combined into a unified report and passed to the AI coaching workflow.

### ☁️ Video Recording Analysis

Upload a recorded presentation and run the complete multimodal analysis pipeline.

The uploaded-video workflow analyzes both visual frames and extracted audio before generating the final report and personalized AI coaching.

---

## ☁️ Deployment

PresentAI Coach is deployed using **Streamlit Community Cloud**.

The deployed application supports:

- Browser-based camera access
- Browser-based microphone access
- Real-time WebRTC presentation sessions
- STUN/TURN connectivity for remote WebRTC communication
- Uploaded-video analysis
- Faster-Whisper speech processing
- ML-based posture analysis
- Gesture and eye-contact analysis
- Gemini-powered agentic coaching

Authenticated TURN connectivity is used to improve WebRTC reliability when direct peer connectivity is unavailable in cloud environments.

Deployment credentials and API keys are stored using secret-management configuration and are **not included in the repository**.

---

## 🧪 Running Tests

Run tests from the project root.

For example:

```powershell
python -m tests.test_posture_scorer
```

```powershell
python -m tests.test_speech_analyzer
```

```powershell
python -m tests.test_session_manager
```

To test the agentic AI coach independently:

```powershell
python -m tests.test_ai_coach
```

The agent test verifies the complete reasoning workflow without requiring a new video analysis.

---

## 📊 Posture Model Development

The posture model was trained using a combination of:

- Real presentation-video samples
- External posture-image samples
- Synthetic samples

The final training dataset contained **998 labeled samples**:

- **BAD:** 522 samples
- **GOOD:** 476 samples

The samples were obtained from:

- **Real presentation video:** 626 samples
- **External/Roboflow data:** 310 samples
- **Synthetic data:** 62 samples

Related samples were grouped during model evaluation to reduce data leakage between training and testing data.

Multiple classifiers were evaluated, including:

- Logistic Regression
- Random Forest
- RBF Support Vector Machine

**Random Forest** was selected for the final implementation.

The selected model was evaluated on an unseen group-safe holdout set.

### Model Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | **74.7%** |
| Macro F1 | **74.5%** |
| ROC-AUC | **80.7%** |

These results should be interpreted as performance on the project's presentation-posture dataset rather than as a universal posture benchmark.

---

## 📈 Evaluation Evidence

Model evaluation evidence is preserved in the `results/` directory.

Generated artifacts include:

- Class distribution visualization
- Dataset source distribution visualization
- Confusion matrix
- ROC curve
- Model performance visualization
- Model metrics CSV
- Classification report CSV

The repository also contains:

```text
generate_submission_results.ipynb
```

This notebook generates the visual evaluation evidence from the saved evaluation model and holdout information without retraining the final production model.

---

## 🔐 Privacy and Data Handling

PresentAI Coach processes presentation media only as required for analysis.

For uploaded presentations, Streamlit temporarily writes the uploaded video to the processing environment. The temporary video is removed after analysis.

For live presentations, browser camera and microphone streams are used during the active WebRTC presentation session.

When PresentAI Coach is run locally, computer-vision and speech-processing operations execute on the local machine.

When the deployed Streamlit application is used, analysis is performed within the deployed application environment.

For AI-generated coaching, analytical results and relevant transcript content are sent to the configured Gemini service so that personalized coaching can be generated.

Users should avoid submitting confidential presentation material when the data-handling terms of the configured external services are unsuitable for that content.

API keys and TURN credentials are loaded through environment or secret configuration and are never intended to be committed to source control.

---

## ⚠️ Current Limitations

PresentAI Coach is a capstone/prototype system and has several known limitations:

- Posture analysis performs best when sufficient upper-body landmarks are visible.
- Laptop-camera angle and partial-body framing can affect posture predictions.
- Eye-contact estimation is a presentation heuristic rather than precise eye tracking.
- Speech-recognition accuracy depends on microphone quality, background noise, and speaker clarity.
- AI coaching quality depends on the accuracy of the analytical evidence provided to the agent.
- Real-time performance depends on available compute resources, browser/network conditions, and camera hardware.
- Presentation-quality metrics are coaching signals and should not be treated as medical, psychological, or ergonomic assessments.

---

## 🚀 Future Improvements

Potential future development includes:

### Visibility-Aware Posture Analysis

Detect whether the camera contains:

- Full torso
- Partial upper body
- Insufficient posture landmarks

and select an appropriate analysis strategy instead of forcing the same model onto every frame.

### Upper-Body Presentation Alignment Model

Develop a dedicated model for realistic laptop/webcam framing where only the head, shoulders, and chest are visible.

This model could operate alongside the existing full-body posture classifier instead of replacing it.

### Personal Calibration

Allow presenters to establish a short neutral baseline before a session to account for:

- Natural shoulder asymmetry
- Camera position
- Camera angle
- Presenter proportions

### Session-to-Session Progress Tracking

Use persistent application state or storage to compare multiple presentation attempts and allow the AI coach to reason about improvement over time.

### Enhanced Coaching Memory

Allow the coach to remember previous presentation goals and evaluate whether the presenter improved those specific areas in later sessions.

---

## 🎯 Project Objective

The goal of PresentAI Coach is not simply to assign presentation scores.

The project explores how **traditional AI/ML systems and modern agentic AI can work together**:

```text
Computer Vision + ML
        ↓
Objective Measurement

Speech Recognition
        ↓
Delivery Evidence

Agentic AI
        ↓
Reasoning + Prioritization

Generative AI
        ↓
Natural Personalized Coaching
```

This hybrid approach allows deterministic analytical components to handle measurable presentation behavior while an agentic reasoning layer transforms those measurements into practical coaching.

---

## 📌 Project Status

**PresentAI Coach — Capstone Complete & Deployed ✅**

Implemented:

- ✅ Real-time browser-based visual presentation analysis
- ✅ Browser camera and microphone capture through WebRTC
- ✅ Uploaded-video analysis
- ✅ ML-based posture scoring
- ✅ Gesture analysis
- ✅ Eye-contact analysis
- ✅ Faster-Whisper speech analysis
- ✅ Session-level performance scoring
- ✅ Streamlit interface
- ✅ LangGraph agentic workflow
- ✅ Gemini-powered personalized coaching
- ✅ AI critic and coaching-quality review
- ✅ Shared Live/Upload analytical architecture
- ✅ Streamlit Community Cloud deployment
- ✅ STUN/TURN support for cloud WebRTC connectivity
- ✅ Model evaluation metrics
- ✅ Evaluation graphs and submission visualizations

Further improvements are planned as future development rather than requirements for the initial capstone release.

---

## 👩‍💻 Author

**Syeda Mahnoor**

Software Engineering Student

---

## 📄 Disclaimer

PresentAI Coach provides automated presentation-practice feedback for educational and self-improvement purposes.

Its posture, gaze, gesture, and speech outputs are computational estimates and should not be interpreted as medical, psychological, accessibility, or ergonomic assessments.