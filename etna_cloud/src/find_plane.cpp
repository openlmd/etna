#include <stdio.h>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

#include <pcl/visualization/cloud_viewer.h>


void showHelp(char *program_name)
{
  std::cout << std::endl;
  std::cout << "Usage: " << program_name << " cloud_filename.pcd" << std::endl;
}

void colorCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr &color_cloud,
                int r, int g, int b)
{
  color_cloud->width = cloud->width;
  color_cloud->height = cloud->height;
  color_cloud->points.resize(cloud->width * cloud->height);
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    color_cloud->points[i].x = cloud->points[i].x;
    color_cloud->points[i].y = cloud->points[i].y;
    color_cloud->points[i].z = cloud->points[i].z;
    color_cloud->points[i].r = r;
    color_cloud->points[i].g = g;
    color_cloud->points[i].b = b;
  }
}

void showCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    colorCloud(cloud, color_cloud, 255, 0, 0);
    viewer.showCloud(color_cloud, "cloud1");
    while (!viewer.wasStopped());
}

void showCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud1,
               const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud2)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud1 (new pcl::PointCloud<pcl::PointXYZRGB>);
    colorCloud(cloud1, color_cloud1, 255, 0, 0);
    viewer.showCloud(color_cloud1, "cloud1");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);
    colorCloud(cloud2, color_cloud2, 0, 0, 255);
    viewer.showCloud(color_cloud2, "cloud2");
    while (!viewer.wasStopped());
}

void showValues(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  std::cout << "Point cloud data: " << cloud->points.size() << std::endl;
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    std::cout << cloud->points[i].x << " "
              << cloud->points[i].y << " "
              << cloud->points[i].z << std::endl;
  }
}


void filterSubsampling(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
  // Voxel Grid Subsampled Filter
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud_in);
  sor.setLeafSize(0.002f, 0.002f, 0.001f);
  sor.filter(*cloud_out);
}

void filterRemoval(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
  // Radius Outlier Removal Filter
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
  outrem.setInputCloud(cloud_in);
  outrem.setRadiusSearch(0.01);
  outrem.setMinNeighborsInRadius(2);
  outrem.filter(*cloud_out);
}


// This is the main function
int matrix_transform(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
  // METHOD #1: Using a Matrix4f for an homogeneous transformation
  Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

  // Define an homogenous transformation matrix
  float theta = M_PI/4; // The angle of rotation in radians
  //transform_1(0,0) = cos (theta);
  //transform_1(0,1) = -sin(theta);
  //transform_1(1,0) = sin (theta);
  //transform_1(1,1) = cos (theta);
  transform_1(0,3) = -1.45; // Translation on the x axis
  transform_1(2,3) = -0.95; // Translation on the z axis

  std::cout << "Transformation matrix\n" << transform_1 << std::endl;

  //Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
  //Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
  //Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
  //Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
  //Eigen::Matrix3d rotationMatrix = q.matrix();

  //Matrix3f m;
  //m = AngleAxisf(0.25*M_PI, Vector3f::UnitX())
  //* AngleAxisf(0.5*M_PI, Vector3f::UnitY())
  //* AngleAxisf(0.33*M_PI, Vector3f::UnitZ());
  //cout << m << endl << "is unitary: " << m.isUnitary() << endl;

  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
  Eigen::Vector2d u(-1,1), v(2,0);
  std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
  std::cout << "Here is mat*u:\n" << mat*u << std::endl;
  std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
  std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
  std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
  std::cout << "Let's multiply mat by itself" << std::endl;
  mat = mat*mat;
  std::cout << "Now mat is mat:\n" << mat << std::endl;

  // METHOD #2: Using a Affine3f. This method is easier and less error prone
  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

  transform_2.translation() << -1.0, 0.0, -0.9;
  // The same rotation matrix as before; tetha radians arround Z axis
  transform_2.rotate(Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));

  // Print the transformation matrix
  std::cout << "\nMethod #2: using an Affine3f\n" << transform_2.matrix() << std::endl;

  pcl::transformPointCloud(*source_cloud, *cloud_out, transform_1);
}


int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if ((argc > 1) && (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *cloud) < 0))  {
    std::cout << "Error loading point cloud " << argv[1] << std::endl << std::endl;
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    showHelp(argv[0]);
    return -1;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZ>);


  filterSubsampling(cloud, cloud_filtered);
  filterRemoval(cloud_filtered, cloud_filtered2);

  std::cerr << "PointCloud before filtering: " << cloud->points.size ()
       << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;

  std::cerr << "PointCloud after subsampling: " << cloud_filtered->width * cloud_filtered->height
       << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;

  std::cerr << "PointCloud after filtering: " << cloud_filtered2->width * cloud_filtered2->height
       << " data points (" << pcl::getFieldsList(*cloud_filtered2) << ")." << std::endl;

  pcl::io::savePCDFileASCII("downsampled.pcd", *cloud_filtered2);



  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true); // Optional
  seg.setModelType(pcl::SACMODEL_PLANE); // Mandatory
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.001);

  seg.setInputCloud(cloud_filtered);
  seg.segment(*inliers, *coefficients);

  // Extract the planar inliers points from the input cloud
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*cloud_plane);


  if (inliers->indices.size () == 0) {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    return (-1);
  }

  float a, b, c, d;
  a = coefficients->values[0];
  b = coefficients->values[1];
  c = coefficients->values[2];
  d = coefficients->values[3];
  std::cerr << "Model coefficients: " << a << " " << b << " " << c << " " << d << std::endl;

  // Plane frame reference
  pcl::PointXYZ pnt0, pnt1, pnt2;
  pcl::PointXYZ vecx, vecy, vecz;
  pnt0.x = d / -a;
  pnt0.y = 0;
  pnt0.z = 0;
  pnt1.x = (c * 10 + d) / -a;
  pnt1.y = 0;
  pnt1.z = 10;
  pnt2.x = (b * 10 + d) / -a;
  pnt2.y = 10;
  pnt2.z = 0;
  printf("Point0: %f %f %f\n", pnt0.x, pnt0.y, pnt0.z);
  printf("Point1: %f %f %f\n", pnt1.x, pnt1.y, pnt1.z);
  printf("Point2: %f %f %f\n", pnt2.x, pnt2.y, pnt2.z);
  vecx.x = pnt1.x - pnt0.x;
  vecx.y = pnt1.y - pnt0.y;
  vecx.z = pnt1.z - pnt0.z;
  //vecx = normalize(pnt1 - pnt0)
  //vecy = normalize(pnt2 - pnt0)
  //vecz = normalize(cross(vecx, vecy))
  //vecy = normalize(cross(vecz, vecx))

  showCloud(cloud_plane);

  pcl::io::savePCDFileASCII("plane_cloud.pcd", *cloud_plane);
  std::cerr << "Saved " << cloud_plane->points.size() << " data points to plane_cloud.pcd." << std::endl;

  //pcl::PointXYZ min_pt;
  //pcl::PointXYZ max_pt;
  //pcl::getMinMax3D(*cloud_plane, min_pt, max_pt);

  pcl::PointXYZ minPt, maxPt;
  pcl::getMinMax3D(*cloud_plane, minPt, maxPt);
  std::cout << "Max x: " << maxPt.x << std::endl;
  std::cout << "Max y: " << maxPt.y << std::endl;
  std::cout << "Max z: " << maxPt.z << std::endl;
  std::cout << "Min x: " << minPt.x << std::endl;
  std::cout << "Min y: " << minPt.y << std::endl;
  std::cout << "Min z: " << minPt.z << std::endl;

  // Executing the transformation
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

  matrix_transform(cloud_filtered, transformed_cloud);

  showCloud(cloud_filtered, transformed_cloud);

  return (0);
}
