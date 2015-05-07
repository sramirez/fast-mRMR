/*
 ***********************************************************************
 * Class: StringTokenizer                                              *
 * By Arash Partow - 2000                                              *
 * URL: http://www.partow.net/programming/stringtokenizer/index.html   *
 *                                                                     *
 * Copyright Notice:                                                   *
 * Free use of this library is permitted under the guidelines and      *
 * in accordance with the most current version of the Common Public    *
 * License.                                                            *
 * http://www.opensource.org/licenses/cpl1.0.php                       *
 *                                                                     *
 * Note: This library has been deprecated in favour of the C++ String  *
 * Toolkit Library (StrTk).                                            *
 * URL: http://www.partow.net/programming/strtk/index.html             *
 *                                                                     *
 ***********************************************************************
*/


#ifndef INCLUDE_STRINGTOKENIZER_H
#define INCLUDE_STRINGTOKENIZER_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>


class StringTokenizer
{
public:
   StringTokenizer(const std::string& _str, const std::string& _delim);
  ~StringTokenizer(){};

   int         countTokens();
   bool        hasMoreTokens();
   std::string nextToken();
   int         nextIntToken();
   double      nextFloatToken();
   std::string nextToken(const std::string& delim);
   std::string remainingString();
   std::string filterNextToken(const std::string& filterStr);

private:

   std::string  token_str;
   std::string  delim;
};

#endif

