//
//  helper.cpp
//  eos
//
//  Created by Vidit Singh on 18/10/16.
//
//

#include "helper.hpp"
#include <iomanip>
#include <sys/stat.h>
#include <iterator>
#include "boost/filesystem.hpp"
#include "opencv2/photo.hpp"
#include "eos/render/render.hpp"
#include <ctime>

#define TRAINING 0

namespace fs = boost::filesystem;
using std::string;
using std::vector;
using cv::Vec4f;
using cv::Vec2f;
using cv::Mat;
using std::cout;
using std::endl;

// create string with leading zeros
string createString(string base, int indx, int lead_zeros = 4){
    std::ostringstream ss;
    ss << base << "_" << std::setfill('0') << std::setw(lead_zeros) << indx << ".jpg";
    return ss.str();
    
}

// check if file exists
bool fileExists (const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}


// convert vector to OpenCV Mat
Mat vvec2mat(vector<cv::Vec4f> vector){
    int cols = vector.size();
    int rows = vector[0].channels;
    Mat mat(rows,cols,CV_32FC1);
    
    for (int i = 0; i < cols ; i++) {
        Mat submat = mat.rowRange(0, rows-1).colRange(i, i);
        Mat temp(vector[i]);
        temp.copyTo(submat);
    }
    
    return mat;
}

// Model View matrix for projection
glm::mat4x4 getModelView(float r_x, float r_y, float r_z)  {
    auto rot_mtx_x = glm::rotate(glm::mat4(1.0f), r_x, glm::vec3{ 1.0f, 0.0f, 0.0f });
    auto rot_mtx_y = glm::rotate(glm::mat4(1.0f), r_y, glm::vec3{ 0.0f, 1.0f, 0.0f });
    auto rot_mtx_z = glm::rotate(glm::mat4(1.0f), r_z, glm::vec3{ 0.0f, 0.0f, 1.0f });
    auto zxy_rotation_matrix = rot_mtx_z * rot_mtx_x * rot_mtx_y;
    auto rotation = glm::quat(zxy_rotation_matrix);
    
    glm::mat4x4 modelview = glm::mat4_cast(rotation);
    modelview[3][0] = 0;
    modelview[3][1] = 0;
    return modelview;
}

// Get pose normalized 2D image from 3D mesh model
Mat frontalImage(eos::render::Mesh mesh, Mat isomap, int width, int height){
    
    eos::render::Texture texture =  eos::render::create_mipmapped_texture(isomap);
    
    
    auto glm_matrix_t0 = glm::transpose(getModelView(0.0,0.0,0.0));
    cv::Mat model_view(4, 4, CV_32FC1, &glm_matrix_t0[0]);
    //    cout << model_view << endl;
    
    
    
    auto glm_matrix_t1 = glm::transpose(glm::ortho(-90.0f, 90.0f, -90.0f, 90.0f));
    cv::Mat projection(4, 4, CV_32FC1, &glm_matrix_t1[0]);
    
    //    cout << projection << endl;
    
    
    std::pair<cv::Mat, cv::Mat> val =  eos::render::render(mesh, model_view, projection , width, height, texture,true,false,false);
    Mat img = val.second ;
    Mat color = val.first;
    
    Mat new_img  = Mat::zeros(color.rows, color.cols, CV_8UC3);
    int roi_size = 5;
    
    vector<cv::Rect> rois;
    for (int height = 2*roi_size; height < color.rows-2*roi_size; height++)
        for (int width = 2*roi_size; width < color.cols-2*roi_size; width++)
        {
            cv::Vec4b rgba = color.at<cv::Vec4b>(height, width );
            new_img.at<cv::Vec3b>(height,width) = cv::Vec3b(rgba.val[0],rgba.val[1],rgba.val[2]);
            if((rgba[0]+rgba[1]+rgba[2])/3 < 1.0f){
                rois.push_back(cv::Rect(width,height,roi_size,roi_size) );
            }
        }
    
    return new_img;
    
}


// Main file to do fitting and getting pose normailzed frontal face
int main(int argc, char *argv[]){
    
    
    vector<Mat> affine_camera_matrices;
    vector<vector<Vec2f>> landmarks_set;
    vector<vector<int>> vertex_ids_set;
    
    int right[16] = {1,2,3,19,20,21,22,23,24,31,32,33,41,42,43,48};
    int left[16] = {7,8,9,25,26,27,28,29,30,35,36,37,38,39,45,46};
    vector<vector<int>> drop_landmarks_set;
    vector<Mat> isomaps;
    vector<fs::path> imagefilename;
    
    string modelfile("/Users/vidit/Semester3/Project/FittingModel/eos/share/sfm_shape_3448.bin");
    string mappingsfile("/Users/vidit/Semester3/Project/FittingModel/eos/share/ibug2did.txt");
    
#if TRAINING
    fs::path parent_dir("/Users/vidit/Semester3/Project/final/compare/train_comb/images_10_3_2");
#else
    fs::path parent_dir("/Users/vidit/Semester3/Project/final/test");
#endif
    
    fs::path images_dir = parent_dir/"images";
    fs::path landmarks_dir = parent_dir/"landmarks";
    
    fs::path models_dir  = parent_dir/"models";
    fs::path frontal_images_dir  = parent_dir/"frontalimages";
    
    fs::create_directory(models_dir);
    fs::create_directory(frontal_images_dir);
    
    vector<fs::directory_entry> landmarks_iter;
    copy(fs::directory_iterator(landmarks_dir), fs::directory_iterator(), back_inserter(landmarks_iter));
    
    for ( vector<fs::directory_entry>::const_iterator it = landmarks_iter.begin(); it != landmarks_iter.end();  ++ it )
    {
        fs::path curr_path =  (*it).path();
        string label = curr_path.stem().string();
        
        std::ifstream single_face_files;
        single_face_files.open(curr_path.string() + "/single_face_images.txt");
        
        fs::create_directory(models_dir/label);
        fs::create_directory(frontal_images_dir/label);
        
        string single_face_file;
        while (std::getline(single_face_files, single_face_file)){
            
            fs::path imagefile = images_dir / label/ single_face_file;
            fs::path landmarksfile = landmarks_dir / label/  fs::path(imagefile.stem().string()+  "_0.pts");
            
           
            
            fs::path outputfile = models_dir/ label/ fs::path(imagefile.stem().string()+"_out");
            
            fs::path frontalimage = frontal_images_dir/ label/ fs::path(imagefile.stem().string()+"_frontal.jpg");
            

            
            if(fileExists(landmarksfile.string())){
                
                string person = "Juan_Carlos_Ferrero_0001";//"Bill_Gates_0005";//"Angelina_Jolie_0010" ; //"Bill_Clinton_0006";
                string test = "/Users/vidit/Semester3/Project/train/images/" + person + ".jpg";
                string ld = "/Users/vidit/Semester3/Project/train/landmarks/" + person + "_0.pts";
                string final = "/Users/vidit/Desktop/report\ figs/"+ person + "_frontal.jpg";
//                std::shared_ptr<model_data> data = fit_model(modelfile, imagefile.string(), landmarksfile.string(), mappingsfile, outputfile.string());
                    std::shared_ptr<model_data> data = fit_model(modelfile, test, ld, mappingsfile, outputfile.string());
                

#if TRAINING
                // Reject landmarks when pose id off
                if (data->yaw_angle < -35) {
                    vector<int> v(right, right + sizeof right / sizeof right[0]);
                    drop_landmarks_set.push_back(v);
                }else if(data->yaw_angle > 35){
                    vector<int> v(left, left + sizeof left / sizeof left[0]);
                    drop_landmarks_set.push_back(v);
                }else{
                    vector<int> v;
                    drop_landmarks_set.push_back(v);
                }
                

                affine_camera_matrices.push_back(data->affine_from_ortho);
                landmarks_set.push_back(data->image_points);
                vertex_ids_set.push_back(data->vertex_indices);
                isomaps.push_back(data->isomap);
                imagefilename.push_back(frontalimage);
                
            }
            
            
            
        }
        
        // Get the shape coefficients by ridge regression
        Helper process;
        vector<float> alphas =  process.regressToFitPoses(eos::morphablemodel::load_model(modelfile), affine_camera_matrices, landmarks_set, vertex_ids_set, drop_landmarks_set);
        
        eos::render::Mesh mesh = eos::morphablemodel::load_model(modelfile).draw_sample(alphas, vector<float>());
        fs::path final_model = parent_dir/"models"/(label+".obj");
        eos::render::write_textured_obj(mesh, final_model.string());
        
        // Do inpainting for empty regions
        clock_t begin = clock();
        for (int i = 0; i < isomaps.size(); i++) {
            Mat mask0 = Mat::zeros(isomaps[i].rows, isomaps[i].cols, CV_8UC1);
            Mat mask1 = Mat::zeros(isomaps[i].rows, isomaps[i].cols, CV_8UC1);
            
            Mat img  = Mat::zeros(isomaps[i].rows, isomaps[i].cols, CV_8UC3);
            
            for (int height = 0; height < isomaps[i].rows; height++)
                for (int width = 0; width < isomaps[i].cols; width++)
                {
                    cv::Vec4b rgba = isomaps[i].at<cv::Vec4b>(height, width );
                    img.at<cv::Vec3b>(height,width) = cv::Vec3b(rgba.val[0],rgba.val[1],rgba.val[2]);
                    if(rgba.val[3] != 255){
                        mask0.at<uint8_t>(height,width) = 255;
                    }
                    
                }
            
            cv::inpaint(img, mask0, img, 0.01f, cv::INPAINT_NS);
            
            
            imwrite(imagefilename[i].string(),frontalImage(mesh, img, 250, 250));
        }
        cout <<"Time taken: " <<  (clock() - begin)/CLOCKS_PER_SEC << endl;

        
        
        drop_landmarks_set.clear();
        affine_camera_matrices.clear();
        landmarks_set.clear();
        vertex_ids_set.clear();
        isomaps.clear();
        imagefilename.clear();
#else
            clock_t begin = clock();
        
            Mat mask = Mat::zeros(data->isomap.rows, data->isomap.cols, CV_8UC1);
            
            Mat img  = Mat::zeros(data->isomap.rows, data->isomap.cols, CV_8UC3);
            
            for (int height = 0; height < data->isomap.rows; height++)
                for (int width = 0; width < data->isomap.cols; width++)
                {
                    cv::Vec4b rgba = data->isomap.at<cv::Vec4b>(height, width );
                    img.at<cv::Vec3b>(height,width) = cv::Vec3b(rgba.val[0],rgba.val[1],rgba.val[2]);
                    if(rgba.val[3] != 255){
                        mask.at<uint8_t>(height,width) = 255;
                    }
                    
                }
            
//            cv::inpaint(img, mask, img, 0.01f, cv::INPAINT_NS);
        
        
            Mat finalimg = frontalImage(data->mesh, img, 250, 250);
            
        cout << "Time Inpaint + frontImage : " << float(clock() - begin)/CLOCKS_PER_SEC << endl;
        
        imwrite(final,finalimg);
        break;
        
//            imwrite(frontalimage.string(),finalimg);
        
    }
}

#endif

}


}


// 3d to 2d regression. Funciton similar to EOS library

vector<float> Helper::regressToFitPoses(eos::morphablemodel::MorphableModel morphable_model, vector<Mat> affine_camera_matrices, vector< vector<Vec2f> > landmarks_set, vector<vector<int>> vertex_ids_set,vector<vector<int>>  drop_landmarks_set, Mat base_face, float lambda){
    
    int num_poses =  affine_camera_matrices.size();
    int num_coeffs_to_fit = morphable_model.get_shape_model().get_num_principal_components();
    int num_landmarks_ =  landmarks_set[0].size();
    
    
    
    Mat Af = Mat::zeros(num_coeffs_to_fit,num_coeffs_to_fit, CV_32FC1);
    Mat bf = Mat::zeros(num_coeffs_to_fit, 1, CV_32FC1);
    
    for (int count = 0 ; count < num_poses; count++) {
        
        Mat affine_camera_matrix = affine_camera_matrices[count];
        vector<Vec2f> landmarks = landmarks_set[count];
        vector<int> vertex_ids = vertex_ids_set[count];
        
        int num_landmarks = static_cast<int>(landmarks.size());
        
        if (base_face.empty())
        {
            base_face = morphable_model.get_shape_model().get_mean();
        }
        
        // $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
        // And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
        Mat V_hat_h = Mat::zeros(4 * num_landmarks, num_coeffs_to_fit, CV_32FC1);
        int row_index = 0;
        for (int i = 0; i < num_landmarks; ++i) {
            Mat basis_rows = morphable_model.get_shape_model().get_normalised_pca_basis(vertex_ids[i]); // In the paper, the not-normalised basis might be used? I'm not sure, check it. It's even a mess in the paper. PH 26.5.2014: I think the normalised basis is fine/better.
            //basisRows.copyTo(V_hat_h.rowRange(rowIndex, rowIndex + 3));
            basis_rows.colRange(0, num_coeffs_to_fit).copyTo(V_hat_h.rowRange(row_index, row_index + 3));
            row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
        }
        // Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
        Mat P = Mat::zeros(3 * num_landmarks, 4 * num_landmarks, CV_32FC1);
        for (int i = 0; i < num_landmarks; ++i) {
            Mat submatrix_to_replace = P.colRange(4 * i, (4 * i) + 4).rowRange(3 * i, (3 * i) + 3);
            affine_camera_matrix.copyTo(submatrix_to_replace);
        }
        // The variances: Add the 2D and 3D standard deviations.
        // If the user doesn't provide them, we choose the following:
        // 2D (detector) standard deviation: In pixel, we follow [1] and choose sqrt(3) as the default value.
        // 3D (model) variance: 0.0f. It only makes sense to set it to something when we have a different variance for different vertices.
        // The 3D variance has to be projected to 2D (for details, see paper [1]) so the units do match up.
        float sigma_squared_2D = std::pow(std::sqrt(3.0f), 2) + std::pow((0.0f), 2);
        Mat Sigma = Mat::zeros(3 * num_landmarks, 3 * num_landmarks, CV_32FC1);
        for (int i = 0; i < 3 * num_landmarks; ++i) {
            Sigma.at<float>(i, i) = 1.0f / std::sqrt(sigma_squared_2D); // the higher the sigma_squared_2D, the smaller the diagonal entries of Sigma will be
        }
        Mat Omega = Sigma.t() * Sigma; // just squares the diagonal
        // The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
        Mat y = Mat::ones(3 * num_landmarks, 1, CV_32FC1);
        for (int i = 0; i < num_landmarks; ++i) {
            y.at<float>(3 * i, 0) = landmarks[i][0];
            y.at<float>((3 * i) + 1, 0) = landmarks[i][1];
            //y.at<float>((3 * i) + 2, 0) = 1; // already 1, stays (homogeneous coordinate)
        }
        // The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
        Mat v_bar = Mat::ones(4 * num_landmarks, 1, CV_32FC1);
        for (int i = 0; i < num_landmarks; ++i) {
            //cv::Vec4f model_mean = morphable_model.get_shape_model().get_mean_at_point(vertex_ids[i]);
            cv::Vec4f model_mean(base_face.at<float>(vertex_ids[i] * 3), base_face.at<float>(vertex_ids[i] * 3 + 1), base_face.at<float>(vertex_ids[i] * 3 + 2), 1.0f);
            v_bar.at<float>(4 * i, 0) = model_mean[0];
            v_bar.at<float>((4 * i) + 1, 0) = model_mean[1];
            v_bar.at<float>((4 * i) + 2, 0) = model_mean[2];
            //v_bar.at<float>((4 * i) + 3, 0) = 1; // already 1, stays (homogeneous coordinate)
            // note: now that a Vec4f is returned, we could use copyTo?
        }
        // Bring into standard regularised quadratic form with diagonal distance matrix Omega
        Mat A = P * V_hat_h; // camera matrix times the basis
        Mat b = P * v_bar - y; // camera matrix times the mean, minus the landmarks.
        //Mat c_s; // The x, we solve for this! (the variance-normalised shape parameter vector, $c_s = [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$
        //int numShapePc = morphableModel.getShapeModel().getNumberOfPrincipalComponents();
        
        if (drop_landmarks_set[count].size() > 0) {
            
            Mat drop_landmark = Mat::ones(3 * num_landmarks, 3 * num_landmarks, CV_32FC1);
            Mat zeros = Mat::zeros(3, 3, CV_32FC1);
            
            for (auto i : drop_landmarks_set[count]) {
                Mat submatrix_to_replace = drop_landmark.colRange(3 * i, (3 * i) + 3).rowRange(3 * i, (3 * i) + 3);
                zeros.copyTo(submatrix_to_replace);
            }
            cout << cv::determinant(drop_landmark) << endl;
            A = drop_landmark * A;
            b = drop_landmark * b;
        }
        
        Mat AtOmegaA = A.t() * Omega * A;
        Af += AtOmegaA;
        bf += -A.t() * Omega.t() * b;
        
    }
    
    const int num_shape_pc = num_coeffs_to_fit;
    Af = Af + num_poses* lambda * Mat::eye(num_shape_pc, num_shape_pc, CV_32FC1);
    
   
    // Solve using OpenCV:
    Mat c_s; // Note/Todo: We get coefficients ~ N(0, sigma) I think. They are not multiplied with the eigenvalues.
    bool non_singular = cv::solve(Af, bf, c_s, cv::DECOMP_SVD); // DECOMP_SVD calculates the pseudo-inverse if the matrix is not invertible.
    return std::vector<float>(c_s);
    
    
    
}
