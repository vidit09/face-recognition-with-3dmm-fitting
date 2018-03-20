//
//  fit-model.h
//  eos
//
//  Created by Vidit Singh on 18/10/16.
//
//

#ifndef fit_model_h
#define fit_model_h

#include "eos/render/utils.hpp"
#include "eos/morphablemodel/PcaModel.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>



class model_data{
    
public:
    std::vector<cv::Vec4f> model_points; // the points in the 3D shape model
    std::vector<int> vertex_indices; // their vertex indices
    std::vector<cv::Vec2f> image_points; // the landmark points in the image
    cv::Mat affine_from_ortho;      // transform matrix
    cv::Mat isomap;                 // texture map
    eos::render::Mesh mesh;          // output mesh
    eos::morphablemodel::PcaModel shape_model;
    float yaw_angle;
    std::vector<float> fitted_coeffs;
    
    model_data(std::vector<cv::Vec4f> model_points, std::vector<int> vertex_indices, std::vector<cv::Vec2f> image_points, cv::Mat affine_from_ortho,eos::render::Mesh mesh , cv::Mat isomap, eos::morphablemodel::PcaModel shape_model, float yaw_angle, std::vector<float> fitted_coeffs){
        
        this->model_points      = model_points;
        this->vertex_indices    = vertex_indices;
        this->image_points      = image_points;
        this->affine_from_ortho = affine_from_ortho;
        this->isomap            = isomap;
        this->mesh              = mesh;
        this->shape_model       = shape_model;
        this->yaw_angle         = yaw_angle;
        this->fitted_coeffs     = fitted_coeffs;
    }
};

std::shared_ptr<model_data> fit_model(std::string modelfilepath, std::string imagefilepath, std::string landmarkfilepath, std::string mappingsfilepath, std::string outputfilepath );

#endif /* fit_model_h */
