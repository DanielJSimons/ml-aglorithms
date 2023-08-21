import numpy as np

class NaiveBayesClassifier:
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Calculate mean, variance, and prior for each class
        self.class_means = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_variances = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_priors = np.zeros(n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_means[idx, :] = X_c.mean(axis=0)
            self.class_variances[idx, :] = X_c.var(axis=0)
            self.class_priors[idx] = X_c.shape[0] / float(n_samples)
        
    def predict(self, X):
        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)
    
    def _predict_sample(self, x):
        posteriors = []
        
        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.class_priors[idx])
            posterior = np.sum(np.log(self._calculate_pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
            
        # Return class with highest posterior
        return self.classes[np.argmax(posteriors)]
        
    def _calculate_pdf(self, class_idx, x):
        mean = self.class_means[class_idx]
        variance = self.class_variances[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator
 
    
# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=762
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=565
    )

    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)
    predictions = nb_classifier.predict(X_test)

    print("Naive Bayes classification accuracy:", accuracy(y_test, predictions))
