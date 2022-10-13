/*
 * Copyright 2020 California  Institute  of Technology (“Caltech”)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if !defined(X_PR_TYPES_H) && defined(MULTI_UAV)
#define X_PR_TYPES_H

#include <memory>

#include <DBow3/Vocabulary.h>

namespace x {
enum DESCRIPTOR_TYPE { ORB = 0 };

class Keyframe;
class Database;

typedef std::shared_ptr<Database> DatabasePtr;
typedef DBoW3::Vocabulary PRVocabulary;
typedef std::shared_ptr<DBoW3::Vocabulary> PRVocabularyPtr;
typedef std::vector<std::shared_ptr<Keyframe>> VectorKeyframe;
typedef std::shared_ptr<Keyframe> KeyframePtr;
typedef cv::Mat VLADVec;

}  // namespace x

#endif