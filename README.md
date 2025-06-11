TrueSight_AI/
│
├── assets/
│   ├── logo.png
│   ├── background.jpg
│   └── sample_images/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── dataset/
│   ├── real/
│   │   ├── real1.jpg
│   │   ├── ...
│   └── fake/
│       ├── fake1.jpg
│       ├── ...
│
├── models/
│   └── ai_detector_state_dict.pth
│
├── outputs/
│   ├── confidence_gauge.png
│   ├── heatmap.png
│   ├── temp_uploaded_image.jpg
│   ├── report_sample.pdf
│   └── videoplayback.mp4 (uploaded videos)
│
├── pages/
│   ├── 1_Try_Demo.py
│   ├── 2_Image_Detection.py
│   ├── 3_Video_Detection.py
│   ├── 4_Detection_History.py
│   └── 5_Report_Generator.py
│
├── utils/
│   ├── model_utils.py
│   ├── report_utils.py
│   └── history_utils.py
│
├── detection_history.json
├── train_model.py
├── README.md
├── requirements.txt
└── Home.py
