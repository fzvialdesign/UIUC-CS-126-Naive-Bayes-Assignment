#include "../../include/core/image_model.h"
#include <filesystem>

namespace naivebayes {

using FloatVector3D = std::vector<std::vector<std::vector<float>>>;
using FloatVector2D = std::vector<std::vector<float>>;

ImageModel::ImageModel() : dimension_(0) {}

int ImageModel::ClassifyImage(const Image& image, bool is_testing) {
  float max_likelihood = CalculateLikelihood(image.GetShades(), Set::kSetOne);
  float likelihood;

  Set set_prediction = Set::kSetOne;

  for (size_t set = 1; set < (size_t)Set::kSize; ++set) {
    likelihood = CalculateLikelihood(image.GetShades(), (Set)set);

    if (likelihood > max_likelihood) {
      max_likelihood = likelihood;
      set_prediction = (Set)set;
    }
  }

  if (set_prediction == image.GetSet() && is_testing) {
    accuracies_.push_back(true);
  } else {
    accuracies_.push_back(false);
  }

  return (int)set_prediction;
}

float ImageModel::RetrieveTotalAccuracy() const {
  float total_accurate = 0.0f;

  for (bool is_accurate : accuracies_) {
    if (is_accurate) {
      ++total_accurate;
    }
  }

  return total_accurate / (float)accuracies_.size();
}

const ImageVector2D &ImageModel::GetImages() const { return images_; }

const FloatVector4D &ImageModel::GetProbabilities() const {
  return probabilities_;
}

const std::vector<float> &ImageModel::GetPriors() const { return priors_; }

const std::vector<bool> &ImageModel::GetAccuracies() const {
  return accuracies_;
}

size_t ImageModel::GetDimension() const { return dimension_; }

std::ostream& operator<<(std::ostream& os, ImageModel& model) {
  os << "SAVE" << std::endl;
  os << model.dimension_ << std::endl;

  for (float prior : model.priors_) {
    os << prior << std::endl;
  }

  for (const FloatVector3D& set : model.probabilities_) {
    for (const FloatVector2D& shade : set) {
      for (const std::vector<float>& points : shade) {
        for (float probability : points) {
          os << probability << std::endl;
        }
      }
    }
  }

  return os;
}

std::istream& operator>>(std::istream& is, ImageModel& model) {
  std::string line;
  std::getline(is, line);

  if (line == "SAVE") {
    model.LoadInSave(is, line);
  } else {
    Set set = (Set)std::stoi(line);
    size_t line_count = 0;

    if (model.probabilities_.empty() || model.priors_.empty()) {
      std::getline(is, line);
      model.dimension_ = line.size();
      ++line_count;

      model.ParseThroughFile(is, line, line_count, set, true);
      model.CalculatePriors();
      model.CalculateProbabilities();
    } else {
      model.ParseThroughFile(is, line, line_count, set, false);
    }
  }

  return is;
}

void ImageModel::LoadInSave(std::istream& is, std::string& line) {
  std::getline(is, line);
  probabilities_ = FloatVector4D((size_t)Set::kSize,
                   FloatVector3D((size_t)Shade::kSize,
                   FloatVector2D(std::stoi(line),
                   std::vector<float>(std::stoi(line)))));
  priors_ = std::vector<float>((size_t)Set::kSize);
  dimension_ = std::stoi(line);

  for (float& prior : priors_) {
    std::getline(is, line);
    prior = std::stof(line);
  }

  for (FloatVector3D& set : probabilities_) {
    for (FloatVector2D& shade : set) {
      for (std::vector<float>& points : shade) {
        for (float& probability : points) {
          std::getline(is, line);
          probability = std::stof(line);
        }
      }
    }
  }
}

float ImageModel::CalculateLikelihood(const ShadeVector2D& shades,
                                      const Set& set) {
  float likelihood = logf(priors_[(size_t)set]);

  for (size_t y_coordinate = 0; y_coordinate < shades.size(); ++y_coordinate) {
    for (size_t x_coordinate = 0; x_coordinate < shades[y_coordinate].size();
         ++x_coordinate) {
      Shade shade = shades[y_coordinate][x_coordinate];

      likelihood += logf(probabilities_[(size_t)set][(size_t)shade]
                                      [y_coordinate][x_coordinate]);
    }
  }

  return likelihood;
}

void ImageModel::ParseThroughFile(std::istream &is, std::string &line,
                                  size_t line_count, Set& set,
                                  bool is_training) {
  StringVector lines;

  while (is.good()) {
    if (line_count == 0) {
      set = (Set)std::stoi(line);
    } else {
      if (line.size() != dimension_) {
        throw std::invalid_argument("Text file images not square");
      }

      lines.push_back(line);
    }

    std::getline(is, line);
    ++line_count;

    if (line_count > dimension_) {
      if (is_training) {
        AddImage(lines, set);
      } else {
        ClassifyImage(Image(lines, set), true);
      }

      lines.clear();
      line_count = 0;
    }
  }
}

void ImageModel::AddImage(const StringVector& lines, const Set& set) {
  for (std::vector<Image>& image_set : images_) {
    if (image_set.back().GetSet() == set) {
      image_set.emplace_back(lines, set);
      return;
    }
  }

  images_.push_back(std::vector<Image>{Image(lines, set)});
}

void ImageModel::CalculatePriors() {
  priors_ = std::vector<float>((size_t)Set::kSize);
  float total_images = 0.0f;
  
  for (const std::vector<Image>& set : images_) {
    total_images += (float)set.size();
    
    float numerator = (kSmoothingValue + (float)set.size());
    priors_[(size_t)set.back().GetSet()] = numerator;
  }
  
  for (float& prior : priors_) {
    prior /= ((float)Shade::kSize * kSmoothingValue + total_images);
  }
}

void ImageModel::CalculateProbabilities() {
  probabilities_ = FloatVector4D((size_t)Set::kSize,
                   FloatVector3D((size_t)Shade::kSize,
                   FloatVector2D(dimension_,
                   std::vector<float>(dimension_))));

  for (const std::vector<Image>& set : images_) {
    for (size_t shade = 0; shade < (size_t)Shade::kSize; ++shade) {
      CalculateForEachPoint(set, shade);
    }
  }
}

void ImageModel::CalculateForEachPoint(const std::vector<Image>& set,
                                       size_t shade) {
  for (size_t y_coordinate = 0; y_coordinate < dimension_; ++y_coordinate) {
    for (size_t x_coordinate = 0; x_coordinate < dimension_; ++x_coordinate) {
      probabilities_[(size_t)set.back().GetSet()][shade]
                    [y_coordinate][x_coordinate]
          = BayesTheorem(set, shade, y_coordinate, x_coordinate);
    }
  }
}

float ImageModel::BayesTheorem(const std::vector<Image>& set, size_t shade,
                               size_t y_coord, size_t x_coord) const {
  float numerator = kSmoothingValue;
  float denominator = (float)Shade::kSize * numerator + (float)set.size();

  for (const Image& image : set) {
    if (image.GetShades()[y_coord][x_coord] == (Shade)shade) {
      ++numerator;
    }
  }

  return numerator / denominator;
}

}  // namespace naivebayes