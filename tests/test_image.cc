#include <catch2/catch.hpp>

#include <core/image.h>

using naivebayes::Image;
using naivebayes::Shade;
using naivebayes::Set;

using naivebayes::ShadeVector2D;
using naivebayes::StringVector;

TEST_CASE("Image constructor") {
  SECTION("Valid arguments") {
    StringVector lines{" + ", " # ", " + "};
    ShadeVector2D shade{{Shade::kNone, Shade::kBlack, Shade::kNone},
                        {Shade::kNone, Shade::kBlack, Shade::kNone},
                        {Shade::kNone, Shade::kBlack, Shade::kNone}};

    REQUIRE(Image(lines, Set::kSetOne).GetShades() == shade);
  }

  SECTION("Edge case (text_form is length one with string length one") {
    StringVector lines{" "};
    ShadeVector2D Shade{{Shade::kNone}};
    
    REQUIRE(Image(lines, Set::kSetOne).GetShades() == Shade);
  }

  SECTION("Invalid argument (text_form is empty)") {
    StringVector lines;
    
    REQUIRE_THROWS_AS(Image(lines, Set::kSetOne), std::invalid_argument);
  }

  SECTION("Invalid argument (text_form strings are empty") {
    StringVector lines{"", "", ""};
    
    REQUIRE_THROWS_AS(Image(lines, Set::kSetOne), std::invalid_argument);
  }

  SECTION("Invalid argument (text_form image is not square") {
    StringVector lines{" + ", " # ", " # ", " + "};
    
    REQUIRE_THROWS_AS(Image(lines, Set::kSetOne), std::invalid_argument);
  }
}