//
// Created by viciopoli on 09.10.22.
//

#include "x/camera_models/camera_none.h"

using namespace x;

void CameraNone::undistort(Feature &feature) {
    feature.setX(feature.getXDist());
    feature.setY(feature.getYDist());
}
