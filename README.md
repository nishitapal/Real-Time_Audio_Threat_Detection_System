# Real-Time_Audio_Threat_Detection_System
Problem Statement
Banks today rely heavily on CCTV cameras for security monitoring. But what happens when a camera fails, is blocked, or simply is not present in every corner of the bank? There is no backup system. A robbery could begin and no one would know until it is too late.

Beyond cameras, there is another gap — no system today actively listens for threatening speech in real time inside a bank. Security guards cannot be everywhere. And traditional keyword matching systems only detect exact words — they miss threats spoken differently or in different languages.

Aim
The aim of this project is to build a real-time, microphone-based threat detection system that can identify aggressive tone and threatening intent from audio — without needing any camera.

Objectives
My objectives were:

One — to capture and preprocess live audio from a microphone in real time.

Two — to detect threatening tone using a deep neural network trained on audio embeddings.

Three — to detect threatening intent from speech using a machine learning classifier.

Four — to combine both layers into one reliable system that minimizes false alarms.

And five — to log every event automatically for audit and legal purposes.

Solution — VoiceGuard
VoiceGuard is a dual-layer audio threat detection pipeline. It continuously listens through a microphone, processes 4-second audio chunks, and passes them through two independent detection layers — a tone model and an intent model — before making a final decision.

Workflow — Step by Step
Step 1 — Audio Capture. The system records audio in 4-second chunks at 16,000 Hz sample rate using the device microphone. It runs continuously with no gaps.

Step 2 — Preprocessing. Each chunk is cleaned using a bandpass filter that keeps only the human speech frequency range — 300 to 3400 Hz. A noise gate then removes low-level background sounds like AC or fan noise. Finally, the audio is normalized to a consistent volume level so the model receives consistent input.

Step 3 — Layer 1: Tone Detection. The cleaned audio is passed into YAMNet — a pre-trained audio embedding model by Google. YAMNet converts the raw audio into 1024 numerical features that represent acoustic properties of the sound. These features are then passed into our custom Deep Neural Network — a 3-layer DNN with 256, 128, and 64 neurons — which predicts the probability that the tone is aggressive or threatening. The threshold is set at 0.90 — meaning only very high-confidence aggressive tones are flagged.

Step 4 — Layer 2: Intent Detection. Simultaneously, the audio is sent to Google Speech Recognition which converts it to text. This text is then analyzed by our intent model — a TF-IDF vectorizer combined with a Logistic Regression classifier — trained on 1298 banking-specific sentences. The model predicts whether the meaning of the sentence is threatening or safe, with a threshold of 0.60.

Step 5 — Combined Decision. The final decision logic works like this: if the intent model says THREAT, an alert is raised. If both tone AND intent say THREAT, it is a high-confidence double-confirmed alert. But if only the tone model says THREAT and intent says SAFE — no alert is raised. This is the key design decision that prevents false alarms on normal aggressive-sounding speech.

Step 6 — Alert and Logging. If a threat is detected, an immediate alert is printed with the transcript, confidence score, reason, and timestamp. A 30-second cooldown prevents repeated alerts for the same incident. Every event — both safe and threat — is saved to voiceguard.log for audit purposes.

Deep Dive — Why These Technologies?
Why YAMNet? YAMNet is pre-trained on millions of audio samples by Google. It already understands acoustic patterns extremely well. Instead of training an audio model from scratch — which would need massive data and compute — we use YAMNet as a feature extractor and train only our DNN on top of it. This gave us strong results with limited data.

Why DNN over other models? YAMNet produces 1024 complex features. Simpler models like Logistic Regression or SVM cannot capture the non-linear patterns in such high-dimensional audio data. Our experiments showed Logistic Regression gave 75%, SVM gave 78%, Random Forest gave 80%, but our DNN achieved 90% accuracy.

Why TF-IDF plus Logistic Regression for intent? Our dataset has 1298 sentences — a medium-sized dataset. Deep learning models like BERT would overfit on this. TF-IDF captures word frequency patterns well, and Logistic Regression is fast, interpretable, and performs excellently on this scale. It gave us 93% accuracy.

Why is intent the primary decision-maker? The tone model was trained on general audio — not real banking conversations. So it sometimes flags normal speech as aggressive. Making intent the primary layer and tone a supporting layer solved this false alarm problem completely.

Results
Our tone model achieved 90% accuracy, 96.1% AUC score, 91% recall, and 85% precision on the test set.

Our intent model achieved 93% overall accuracy. For threat sentences specifically — 92% precision and 94% recall — meaning it catches 94 out of every 100 real threats.

We also identified an overfitting issue in the intent model. Cross-validation showed high variance — scores ranging from 69.9% to 100% across folds, with a standard deviation of 0.13. We fixed this by reducing n-gram range from (1,3) to (1,2), reducing max features from 3000 to 2000, and reducing the regularization parameter C from 5 to 0.5. This made the model more generalizable.
