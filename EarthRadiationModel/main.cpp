//
//  main.cpp
//  EarthRadiationModel
//
//  Created by 李桢 on 31/03/2018.
//  Copyright © 2018 李桢. All rights reserved.
//

#include <iostream>
#include "GEarthRadiationModel.h"

using namespace gfc;

int main(int argc, const char * argv[])
{
    GEarthRadiationModel mymodel;
    mymodel.createFluxModel();
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}
