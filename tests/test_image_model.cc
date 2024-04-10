#include <catch2/catch.hpp>

#include <core/image_model.h>
#include <fstream>

using naivebayes::Image;
using naivebayes::ImageModel;

using naivebayes::Shade;
using naivebayes::Set;
using naivebayes::FloatVector4D;

using naivebayes::StringVector;

// Until a solution is found, the only option is to use the absolute filepath
TEST_CASE("Model constructor") {
  ImageModel model = ImageModel();

  REQUIRE(model.GetDimension() == 0);
  REQUIRE(model.GetImages().empty());
  REQUIRE(model.GetPriors().empty());
  REQUIRE(model.GetProbabilities().empty());
  REQUIRE(model.GetAccuracies().empty());
}

TEST_CASE("Handling training and saves") {
  std::string filepath = "C:\\Users\\cesco\\OneDrive\\Documents\\School\\UIUC\\"
                         "2020-2021\\Spring 2021\\CS 126\\Cinder\\my-projects\\"
                         "naive-bayes-fvial2\\tests\\test-data\\"
                         "testimagesandlabels.txt";
  size_t dimension = 1;
  ImageModel model = ImageModel();

  std::ifstream input_file(filepath);
  if (input_file.is_open()) {
    input_file >> model;

    input_file.close();
  }

  FloatVector4D probabilities = model.GetProbabilities();

  SECTION("Successful text file conversion (not save)") {
    size_t valid_size = 2;

    REQUIRE(model.GetImages().size() == valid_size);
    REQUIRE(model.GetImages().front().size() == valid_size);
  }

  SECTION("Successful priors calculation") {
    std::vector<float> priors((size_t)Set::kSize);
    priors[(size_t)Set::kSetOne] = 3.0f / 5.0f;
    priors[(size_t)Set::kSetTwo] = 2.0f / 5.0f;

    REQUIRE(model.GetPriors() == priors);
  }

  SECTION("Successful probabilities calculation") {
    float none = 0.0f;
    float set_one = 1.0f / 2.0f;
    float set_two_none = 1.0f / 3.0f;
    float set_two_black = 2.0f / 3.0f;

    FloatVector4D actual = {{{{set_one}}, {{set_one}}},
                            {{{set_two_none}}, {{set_two_black}}},
                            {{{none}}, {{none}}}, {{{none}}, {{none}}},
                            {{{none}}, {{none}}}, {{{none}}, {{none}}},
                            {{{none}}, {{none}}}, {{{none}}, {{none}}},
                            {{{none}}, {{none}}}, {{{none}}, {{none}}}};

    for (size_t set = 0; set < (size_t)Set::kSize; ++set) {
      for (size_t shade = 0; shade < (size_t)Shade::kSize; ++shade) {
        for (size_t y_coordinate = 0; y_coordinate < dimension;
             ++y_coordinate) {
          for (size_t x_coordinate = 0; x_coordinate < dimension;
               ++x_coordinate) {
            REQUIRE(probabilities[set][shade][y_coordinate][x_coordinate]
                    == actual[set][shade][y_coordinate][x_coordinate]);
          }
        }
      }
    }
  }

  SECTION("Saving and loading a file") {
    std::string save_filepath = "C:\\Users\\cesco\\OneDrive\\Documents\\"
                                "School\\UIUC\\2020-2021\\Spring 2021\\CS 126\\"
                                "Cinder\\my-projects\\naive-bayes-fvial2\\"
                                "tests\\test-data\\testsave.txt";
    ImageModel save = ImageModel();

    std::ofstream output_file(save_filepath);
    if (output_file.is_open()) {
      output_file << model << std::endl;

      output_file.close();
    }

    input_file = std::ifstream(save_filepath);
    if (input_file.is_open()) {
      input_file >> save;

      input_file.close();
    }

    REQUIRE(model.GetPriors() == save.GetPriors());

    FloatVector4D match = save.GetProbabilities();
    for (size_t set = 0; set < dimension; ++set) {
      for (size_t shade = 0; shade < dimension; ++shade) {
        for (size_t x_coordinate = 0; x_coordinate < dimension;
             ++x_coordinate) {
          for (size_t y_coordinate = 0; y_coordinate < dimension;
               ++y_coordinate) {
            REQUIRE(probabilities[set][shade][x_coordinate][y_coordinate]
                    == match[set][shade][x_coordinate][y_coordinate]);
          }
        }
      }
    }
  }

  SECTION("Bad file") {
    std::string bad_file = "C:\\Users\\cesco\\OneDrive\\Documents\\"
                                "School\\UIUC\\2020-2021\\Spring 2021\\CS 126\\"
                                "Cinder\\my-projects\\naive-bayes-fvial2\\"
                                "tests\\test-data\\faultyimagesandlabels.txt";

    input_file = std::ifstream(bad_file);
    if (input_file.is_open()) {
      REQUIRE_THROWS_AS(input_file >> model, std::invalid_argument);

      input_file.close();
    }
  }
}

// Both successful classification and accuracies confirm correct math
TEST_CASE("Handling classification and accuracy") {
  std::string test_small = "C:\\Users\\cesco\\OneDrive\\Documents\\School\\"
                           "UIUC\\2020-2021\\Spring 2021\\CS 126\\Cinder\\"
                           "my-projects\\naive-bayes-fvial2\\tests\\"
                           "test-data\\accuracyimagesandlabels.txt";
  ImageModel small = ImageModel();

  std::ifstream input_file(test_small);
  if (input_file.is_open()) {
    input_file >> small;

    input_file.close();
  }

  input_file = std::ifstream(test_small);
  if (input_file.is_open()) {
    input_file >> small;

    input_file.close();
  }


  SECTION("Successful classification") {
    StringVector lines = {" ++", "+ +", "++ "};

    REQUIRE(small.ClassifyImage(Image(lines, Set::kSetOne), false) == 0);
  }

  SECTION("Accuracy rate of 70% or higher") {
    std::string save_filepath = "C:\\Users\\cesco\\OneDrive\\Documents\\"
                                "School\\UIUC\\2020-2021\\Spring 2021\\CS 126\\"
                                "Cinder\\my-projects\\naive-bayes-fvial2\\"
                                "data\\trainedmodelsave.txt";
    std::string test_filepath = "C:\\Users\\cesco\\OneDrive\\Documents\\"
                                "School\\UIUC\\2020-2021\\Spring 2021\\CS 126\\"
                                "Cinder\\my-projects\\naive-bayes-fvial2\\"
                                "data\\newimagesandlabels.txt";

    ImageModel model = ImageModel();

    input_file = std::ifstream(save_filepath);
    if (input_file.is_open()) {
      input_file >> model;

      input_file.close();
    }

    input_file = std::ifstream(test_filepath);
    if (input_file.is_open()) {
      input_file >> model;

      input_file.close();
    }

    REQUIRE(model.RetrieveTotalAccuracy() >= 0.7f);
  }

  SECTION("Successful calculations of accuracies") {
    std::vector<bool> expected = {true, true, true, true, true,
                                  true, true, true, true};

    REQUIRE(small.GetAccuracies() == expected);
  }
}
