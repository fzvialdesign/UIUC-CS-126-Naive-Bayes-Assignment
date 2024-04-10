#include <iostream>

#include <core/image_model.h>
#include <fstream>

int main() {
  std::string filepath = "C:\\Users\\cesco\\OneDrive\\Documents\\School\\UIUC\\"
                         "2020-2021\\Spring 2021\\CS 126\\Cinder\\my-projects\\"
                         "naive-bayes-fvial2\\data\\"
                         "trainingimagesandlabels.txt";

  naivebayes::ImageModel model = naivebayes::ImageModel();
  std::ifstream input_file(filepath);
  if (input_file.is_open()) {
    input_file >> model;

    input_file.close();
  }

  filepath = "C:\\Users\\cesco\\OneDrive\\Documents\\School\\UIUC\\2020-2021\\"
             "Spring 2021\\CS 126\\Cinder\\my-projects\\naive-bayes-fvial2\\"
             "data\\trainedmodelsave.txt";

  std::ofstream output_file(filepath);
  if (output_file.is_open()) {
    output_file << model << std::endl;

    output_file.close();
  }

  std::cout << "Model successfully trained." << std::endl;

  filepath = "C:\\Users\\cesco\\OneDrive\\Documents\\School\\UIUC\\2020-2021\\"
             "Spring 2021\\CS 126\\Cinder\\my-projects\\naive-bayes-fvial2\\"
             "data\\newimagesandlabels.txt";

  input_file = std::ifstream (filepath);
  if (input_file.is_open()) {
    input_file >> model;

    input_file.close();
  }

  std::cout << "Classifier accuracy: " << model.RetrieveTotalAccuracy() <<
      std::endl;
  return 0;
}
