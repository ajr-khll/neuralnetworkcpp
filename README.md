# neuralnetworkcpp
A basic neural network for the MNIST Dataset, coded from scratch in c++.


The csv_loader module reads data in from the CSV's accessible on kaggle.com, and load them into a custom-defined Matrix class.

The Matrix class is designed with methods that simplify the forward and backward propagation processes; it contains several
operator overloads and methods that allow for linear algebra and calculus functions with minimal verbosity in main.cpp.

This model was able to achieve ~92% accuracy. However, I believe that with further optimization, ~95-97% is achievable without the use of CNNs. 
