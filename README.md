# IDAI201--2505654--JaineePatel-ParkvissionAI
AI-powered parking lot occupancy detector with real-time slot availability insights.

Problem Understanding

The system takes a parking lot image as input and outputs the status of each slot as occupied or empty, along with availability counts and a recommendation. Instead of classifying the whole image, the approach used is crop-based classification, where the image is divided into a grid of slots and each one is classified individually. Key challenges considered include shadows, lighting variation, partial occlusion, and different weather conditions.

Dataset and Preprocessing

The dataset consists of two classes, occupied and empty, with 400 images per class in each of the training, validation, and test splits, split in a 70/15/15 ratio. All images were resized to 224x224 pixels and augmented using rotation, flipping, and brightness adjustment to improve generalization.

Model Development

The model uses MobileNetV2 with pretrained ImageNet weights as the base, with its layers frozen. On top of this, a GlobalAveragePooling layer, a dropout layer, and two dense layers were added, ending in a sigmoid output for binary classification. It was trained using the Adam optimizer and binary crossentropy loss. The final model achieved 96 percent accuracy on the test set, with strong precision and recall for both classes. The confusion matrix and training curves are saved in the models folder as training_history.png.

System Logic and Insights

After classifying all slots in an image, the system calculates the total number of slots, how many are occupied, and how many are available, along with the occupancy percentage. Based on this percentage, the parking lot is categorized as low, moderate, or high congestion, and a recommendation is generated telling the user whether to proceed to park or try another location.

Web App

The web app is built using Streamlit. Users can upload a parking lot image and adjust the slot grid to match the layout. The app then displays the original and annotated images side by side, with green boxes marking empty slots and red boxes marking occupied ones. It also shows live metrics for total, occupied, and available slots, the occupancy percentage, congestion level, and recommendation, along with an option to download the annotated result.


Testing and Deployment

The model was tested on a held-out test set that was not used during training, as well as additional unseen images under different lighting conditions. Some limitations were observed with heavy shadows and extreme camera angles. The project was deployed by pushing the code, trained model, and requirements file to GitHub, then connecting the repository to Streamlit Community Cloud to generate a live public link.

