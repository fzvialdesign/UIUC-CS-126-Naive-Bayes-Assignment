#include "../../include/core/image.h"

namespace naivebayes {

Image::Image(const ShadeVector2D& shades) : shades_(shades), set_(Set::kSize) {
  if (shades.empty()) {
    throw std::invalid_argument("Image (shades form) is empty");
  }
}

Image::Image(const StringVector& lines, const Set& set) : set_(set) {
  if (lines.empty()) {
    throw std::invalid_argument("Image (text form) is empty");
  }

  PopulateShades(lines);
}

const ShadeVector2D &Image::GetShades() const { return shades_; }

const Set &Image::GetSet() const { return set_; }

void Image::PopulateShades(const StringVector& lines) {
  std::vector<Shade> line_shades;

  for (const std::string& line : lines) {
    if (line.size() != lines.size()) {
      throw std::invalid_argument("Image (text form) is not square");
    }

    for (char pixel : line) {
      if (pixel == kBlackShade || pixel == kGreyShade) {
        line_shades.push_back(Shade::kBlack);
      } else {
        line_shades.push_back(Shade::kNone);
      }
    }

    shades_.push_back(line_shades);
    line_shades.clear();
  }
}

}  // namespace naivebayes
