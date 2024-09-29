import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from pdfminer.high_level import extract_text

# Function to extract text from a PDF file
def extract_resume_text(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Global variable for vectorizer
vectorizer = TfidfVectorizer()

# Function to preprocess and embed resumes
def preprocess_and_embed(resume_texts):
    embeddings = vectorizer.fit_transform(resume_texts).toarray()
    return embeddings

# Function to evaluate different SVR kernels
def evaluate_svr_kernels(X_train, X_test, y_train, y_test):
    kernels = ['linear', 'poly', 'rbf']
    best_kernel = None
    best_score = float('-inf')

    for kernel in kernels:
        svr = SVR(kernel=kernel)
        svr.fit(X_train, y_train)
        predictions = svr.predict(X_test)
        
        # Calculate evaluation metrics
        r2 = r2_score(y_test, predictions)
        print(f"Kernel: {kernel}, R² Score: {r2:.4f}")
        
        # Compare R² scores
        if r2 > best_score:
            best_score = r2
            best_kernel = kernel
            
    print(f"Best kernel is: {best_kernel}")
    return best_kernel

# Main function to upload, train, and rank resumes
def rank_resumes():
    # Input: Number of resumes to train
    num_resumes = int(input("Enter the number of resumes to train on: "))
    
    # Ensure at least two resumes for effective training
    if num_resumes < 2:
        print("Please enter at least 2 resumes for training.")
        return
    
    resume_paths = []
    hr_scores = []
    
    # Input: Resumes and HR scores
    print(f"Enter the {num_resumes} resume file names (with .pdf extension) and corresponding HR scores.")
    for i in range(num_resumes):
        resume_file = input(f"Enter the file name for resume {i+1} (e.g., resume.pdf): ")
        resume_paths.append(resume_file)
        
        # Validate HR score input
        while True:
            try:
                hr_score = float(input(f"Enter the HR score for resume {i+1} (e.g., 85.0): "))
                hr_scores.append(hr_score)
                break  # Exit loop if input is valid
            except ValueError:
                print("Invalid input! Please enter a valid number for HR score.")
    
    # Extract text from all provided resume PDFs
    resume_texts = [extract_resume_text(path) for path in resume_paths]
    
    # Preprocess and embed resumes
    resume_embeddings = preprocess_and_embed(resume_texts)
    
    # Standardize embeddings
    scaler = StandardScaler()
    resume_embeddings = scaler.fit_transform(resume_embeddings)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(resume_embeddings, hr_scores, test_size=0.2, random_state=42)
    
    # Evaluate different SVM kernels and get the best one
    best_kernel = evaluate_svr_kernels(X_train, X_test, y_train, y_test)
    
    # Now, use the best kernel for ranking new resumes
    best_svr = SVR(kernel=best_kernel)
    best_svr.fit(X_train, y_train)
    
    # Input: Folder path for new resumes to rank
    folder_path = input("Enter the folder path containing new resumes (with .pdf extension): ")
    
    # Collect all PDF files from the folder
    test_resume_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not test_resume_paths:
        print("No PDF resumes found in the specified folder.")
        return
    
    # Process test resumes to rank
    test_resume_texts = []
    for path in test_resume_paths:
        text = extract_resume_text(path)
        if text:  # Only append if extraction was successful
            test_resume_texts.append(text)
    
    # If no resumes could be processed, exit
    if not test_resume_texts:
        print("No valid resumes found for ranking.")
        return

    # Use the same vectorizer fitted on training data for the test data
    test_embeddings = vectorizer.transform(test_resume_texts)
    test_embeddings = scaler.transform(test_embeddings)
    
    # Predict and rank
    predicted_scores = best_svr.predict(test_embeddings)
    ranked_resumes = sorted(zip(test_resume_paths, predicted_scores), key=lambda x: -x[1])
    
    # Print the ranking
    print("\nResume Ranking based on predicted scores:")
    for rank, (resume, score) in enumerate(ranked_resumes, start=1):
        print(f"{rank}: {resume} - Predicted Score: {score:.4f}")

# Example Usage
if __name__ == "__main__":
    rank_resumes()

