#pragma once

#include "enums.h"

#include <string>
#include <vector>

namespace naivebayes {

using ShadeVector2D = std::vector<std::vector<Shade>>;
using StringVector = std::vector<std::string>;

/**
 * Holds a 2D vector of pixel shade values formed from a text form of the
 * image in the constructor as well as the set of said image.
 */
class Image {
 public:
   /**
    * Loads in an image as pixel shades, storing them in a 2D vector (used in
    * naive_bayes_app.cc).
    * @param shade The image as pixel shades
    */
   Image(const ShadeVector2D& shades);

   /**
    * Loads in an image as text lines and image set using a private method in
    * this constructor which retrieves and stores all shade values for the
    * pixels in the image, storing them in a 2D vector.
    * @param lines The image as text lines
    * @param set The image set the image belongs to
    */
   Image(const StringVector& lines, const Set& set);

   const ShadeVector2D &GetShades() const;

   const Set &GetSet() const;

 private:
   const char kBlackShade = '#';
   const char kGreyShade = '+';

   /**
    * Populates the pixel shades 2D vector based on the text lines of the
    * processed image. Pixel shades are represented by the Shade enum.
    * @param lines The image as text lines
    */
   void PopulateShades(const StringVector& lines);

   ShadeVector2D shades_;
   Set set_;
};

}  // namespace naivebayes
