//
//  helper.hpp
//  eos
//
//  Created by Vidit Singh on 18/10/16.
//
//

#ifndef helper_hpp
#define helper_hpp

#include "fit-model.h"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtc/matrix_transform.hpp"


class Helper{
    
public:
    std::vector<float> regressToFitPoses(eos::morphablemodel::MorphableModel morphable_model, std::vector<cv::Mat> affine_camera_matrices, std::vector< std::vector<cv::Vec2f> > landmarks, std::vector<std::vector<int>> vertex_ids, std::vector<std::vector<int>>  drop_landmarks_set, cv::Mat base_face=cv::Mat(), float lambda=3.0f);
    
   
    
};

#endif /* helper_hpp */
