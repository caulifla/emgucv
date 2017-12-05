/* Calibrating a camera using emgucv
 * reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
 */

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace emguCV {
   static class Calibration {
      public static void Calibrate () {
         objectPoints = new MCvPoint3D32f[frames][];
         imgPoints = new VectorOfPointF[frames];
         rVecs = new Mat[frames];
         tVecs = new Mat[frames];
         width = 5;
         height = 5;
         patternSize = new Size (width, height);
         var files = Directory.GetFiles (@"--location--");
         for (int k = 0; k < frames; k++) {
            grayFrame = CvInvoke.Imread (files[k], ImreadModes.Grayscale);
            found = CvInvoke.FindChessboardCorners (grayFrame, patternSize, cornerPoints, CalibCbType.AdaptiveThresh);
            if (found) {
               //read more about its use and last 3 arguments
               CvInvoke.CornerSubPix (grayFrame, cornerPoints, new Size (11, 11), new Size (-1, -1), new MCvTermCriteria (30, 0.1));
               var objectList = new List<MCvPoint3D32f> ();
               // populating real world coordinates of the chess board corners
               for (int i = 0; i < patternSize.Height; i++)
                  for (int j = 0; j < patternSize.Width; j++)
                     objectList.Add (new MCvPoint3D32f (j * squareSize, i * squareSize, 0.0f));
               objectPoints[k] = objectList.ToArray ();
               imgPoints[k] = cornerPoints;
            }
         }

         // Calibrate Camera
         double error = CalibrateCamera (objectPoints, imgPoints.Select (a => a.ToArray ()).ToArray (), grayFrame.Size, cameraMatrix, distCoeffs, CalibType.RationalModel, new MCvTermCriteria (30, 0.1), out rVecs, out tVecs);

         // Get Optimal new Camera Matrix
         var imgSize = CvInvoke.Imread (files[4], ImreadModes.Grayscale).Size;
         Rectangle ROI = new Rectangle ();
         newMatrix = CvInvoke.GetOptimalNewCameraMatrix (cameraMatrix, distCoeffs, imgSize, 1, imgSize, ref ROI);

         Mat dupFrame = grayFrame.Clone ();

         // Undistort
         CvInvoke.Undistort (grayFrame, dupFrame, newMatrix, distCoeffs);
         var frame = dupFrame.Clone ();
         CvInvoke.Imwrite ("undistorted.png", frame);

         // Region of Interest
         //var buffer_im = _frame.ToImage<Bgr, byte> ();
         //buffer_im.ROI = ROI;
         //Image<Bgr, byte> cropped_im = buffer_im.Copy ();
         //cropped_im.Save ("cropped.png");

         // Drawing detected chessboard corners
         CvInvoke.DrawChessboardCorners (grayFrame, patternSize, cornerPoints, found);

         //CvInvoke.Imwrite ("chessboard.png", _frame);
         CvInvoke.Imwrite ("distorted.png", grayFrame);
         Console.WriteLine (MeanError ());
      }

      // EMGU's calibrate camera method has a bug.
      // Refer this case: https://stackoverflow.com/questions/33127581/how-do-i-access-the-rotation-and-translation-vectors-after-camera-calibration-in
      public static double CalibrateCamera (MCvPoint3D32f[][] objectPoints, PointF[][] imagePoints, Size imageSize, IInputOutputArray cameraMatrix, IInputOutputArray distortionCoeffs, CalibType calibrationType, MCvTermCriteria termCriteria, out Mat[] rotationVectors, out Mat[] translationVectors) {
         System.Diagnostics.Debug.Assert (objectPoints.Length == imagePoints.Length, "The number of images for objects points should be equal to the number of images for image points");
         int imageCount = objectPoints.Length;
         using (VectorOfVectorOfPoint3D32F vvObjPts = new VectorOfVectorOfPoint3D32F (objectPoints))
         using (VectorOfVectorOfPointF vvImgPts = new VectorOfVectorOfPointF (imagePoints)) {
            double reprojectionError;
            using (VectorOfMat rVecs = new VectorOfMat ())
            using (VectorOfMat tVecs = new VectorOfMat ()) {
               reprojectionError = CvInvoke.CalibrateCamera (vvObjPts, vvImgPts, imageSize, cameraMatrix, distortionCoeffs, rVecs, tVecs, calibrationType, termCriteria);
               rotationVectors = new Mat[imageCount];
               translationVectors = new Mat[imageCount];
               for (int i = 0; i < imageCount; i++) {
                  rotationVectors[i] = new Mat ();
                  using (Mat matR = rVecs[i])
                     matR.CopyTo (rotationVectors[i]);
                  translationVectors[i] = new Mat ();
                  using (Mat matT = tVecs[i])
                     matT.CopyTo (translationVectors[i]);
               }
            }
            return reprojectionError;
         }
      }

      static double MeanError () {
         double error = 0;
         for (int i = 0; i < objectPoints.Length; i++) {
            var imgpoints = CvInvoke.ProjectPoints (objectPoints[i], rVecs[i], tVecs[i], cameraMatrix, distCoeffs);
            error += CvInvoke.Norm (imgPoints[i], new VectorOfPointF (imgpoints), NormType.L2) / imgpoints.Length;
         }
         return error / objectPoints.Length;
      }

      static int squareSize = 1; //single chessboard square in mm
      static int frames = 5;
      static VectorOfPointF[] imgPoints;
      static MCvPoint3D32f[][] objectPoints;
      private static Mat grayFrame = new Mat ();
      static int width; //width of chessboard no. squares in width - 1
      static int height; // heght of chess board no. squares in heigth - 1
      private static Size patternSize;  //size of chess board to be detected
      static VectorOfPointF cornerPoints = new VectorOfPointF (); //corners found from chessboard

      static readonly Mat cameraMatrix = new Mat (3, 3, DepthType.Cv64F, 1);
      static readonly Mat distCoeffs = new Mat (8, 1, DepthType.Cv64F, 1);

      static bool found;

      static Mat newMatrix;

      static Mat[] rVecs, tVecs;
   }
}
