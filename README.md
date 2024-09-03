# WODex 
## Description:
**WODex_b** is a child-friendly application designed to help young users explore and learn about the world through their smartphones. 
This project is inspired by the curiosity of children and aims to provide an interactive and educational experience.
**WODex** is a distributed system application designed to provide an educational and interactive experience for children. The application leverages several advanced technologies to manage and serve its AI models efficiently. The app features three major functionalities: 

The backend infrastructure uses **TorchServe** to handle the deployment and management of AI models, ensuring that model inference is efficient and scalable. The task queue is managed by **Celery**, which handles asynchronous task processing, allowing the application to manage workloads effectively. The overall system is orchestrated using **Django**, which serves as the web framework to handle the application's logic and user interactions.
## Features:
### Face Recognition:
- Face Clustering: Automatically groups similar faces together, allowing children to recognize and categorize faces of family members and friends.
- Phone Integration: Links recognized faces with contact information stored on the phone, providing a personalized experience.
### Object Pokedex:
- Image-Based Object Recognition: Inspired by the Pokedex from the popular Pokémon series, this feature allows children to identify objects in the real world by capturing photos.
- Text Input Recognition: Enables children to identify objects through textual input, broadening their learning experience.
- Object Clustering: Groups similar objects together, helping children understand categories and relationships between different items.
### OCR (Optical Character Recognition)
- Text Recognition in Images: Detects and reads text from images, helping children learn to read and recognize words from their surroundings.
This project is designed with a focus on child safety and ease of use, providing an engaging way for kids to interact with and learn from the world around them.


### Installation Guide:
#### clone repository:
```
git clone https://github.com/your-repo/WODex_b.git`
cd WODex_b
```

#### create a Conda environment:
```
conda create --name WODex_b python=3.8
conda activate WODex_b
pip install -r requirements.txt
```

The system operates across three ports, each dedicated to specific services:
**Celery**: Manages task queues, ensuring that tasks are processed in the background efficiently.
```
bash celery.bash
```

**Torchserve**: Manages the AI models, with two modes of operation:
- Build (first time setup): 
Initializes the environment and prepares models for deployment.
```
bash torchserve.bash -b
```
- Run (subsequent runs): Starts the TorchServe service to handle model inference. Command: torchserve.bash
```
bash torchserve.bash
```

**Django**:Manages the web application logic and serves the user interface.
- Download Model Weights (first time only): Downloads the necessary AI model weights
```
bash server.bash -b
```
- Run Server: Starts the Django development server to handle user requests. Command: python3 manage.py runserver
```
bash server.bash
```
