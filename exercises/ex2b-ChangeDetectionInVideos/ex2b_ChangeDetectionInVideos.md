# Introduction

The goal of this exercise is to create a small program for real-time change detection using OpenCV.

# Learning Objectives

After completing this exercise, the student should be able to do the following:

-  Use OpenCV to access a web-camera or the camera or a mobile phone.
-  Use the OpenCV function \texttt{cvtColor} to convert from color to gray scale,
-  Convert images from integer to floating point using the \texttt{astype} function.
-  Convert image from floating point to uint8 using the \texttt{astype} function.
-  Compute a floating point absolute difference image between a new and a previous image.
-  Compute the frames-per-second of an image analysis system.
-  Show text on an image using the OpenCV function \texttt{putText}.
-  Display an image using the OpenCV function \texttt{imshow}.
-  Implement and test a change detection program.
-  Update a background image using a linear combination of the previous background image and a new frame.
-  Compute a binary image by thresholding an absolute difference image.
-  Compute the total number of changed pixels in a binary image.
-  Implement a simple decision algorithm that is based on counting the amount of changed pixels in an image.


# Installing Python packages

In this exercise, we will be using the popular [*OpenCV*](https://opencv.org/) library to perform real-time image analysis.

We will use the virtual environment from the previous exercise (`course02502`). Start an **Anaconda prompt** and do:

```
activate course02502
conda install -c conda-forge opencv
```

You might also need to install **Numpy**:

```
conda install -c anaconda numpy
```

# Exercise data and material

The data and material needed for this exercise can be found here:
(https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex2b-ChangeDetectionInVideos/data/)


Start by creating an exercise folder where you keep your data, Python scripts or Notebooks. Download the data and material and place them in this folder.


# OpenCV program for image differencing

In the exercise material, there is an OpenCV program, `Ex2b-ChangeDetectionInVideosExercise.py`, that:

\begin{enumerate}
  \item Connects to a camera
  \item Acquire images\footnote{Note that we sometimes refers to an image as a \textit{frame}.}, converts them to gray-scale and after that to floating point images
  \item Computes a difference image between a current image and the previous image.
  \item Computes the frames per second (fps) and shows it on an image.
  \item Shows images in windows.
  \item Checks if the key q has been pressed and stops the program if it is pressed.
\end{enumerate}

It is possible to use a mobile phone as a remote camera by following the instructions in Appendix~\ref{app:droidcam}.

\subsubsection*{Exercise \theexno}
\addtocounter{exno}{1}

Run the program \texttt{Ex2b-ChangeDetectionInVideosExercise.py} and see if shows the expected results? Try to move your hands in front of the camera and try to move the camera and see the effects on the difference image.


\subsubsection*{Exercise \theexno}
\addtocounter{exno}{1}

Identify the steps above in the program. What function is used to convert a color image to a gray-scale image?


\section*{Change detection by background subtraction}

The goal of this exercise, is to modify the program in:\\
\texttt{Ex2b-ChangeDetectionInVideosExercise.py}
, so it will be able to raise an alarm if significant changes are detected in a video stream.

The overall structure of the program should be:

\begin{itemize}
  \item Connect to camera
  \item Acquire a background image, convert it to grayscale and then to floating point
  \item Start a loop:
  \begin{enumerate}
    \item Acquire a new image, convert it to grayscale and then to floating point ($I_\text{new}$).
    \item Computes an absolute difference image between the new image and the background image.
    \item Creates a binary image by applying a threshold, T, to the difference image.
    \item Computes the total number of foreground, F, pixels in the foreground image.
    \item Decides if an alarm should be raised if F is larger than an alert value, A.
    \item If an alarm is raised, show a text on the input image. For example \texttt{Change Detected!}.
    \item Shows the input image, the backround image, the difference image, and the binary image. The binary image should be scaled by 255.
    \item Updates the background image ($I_\text{background}$) using: $$I_\text{background} = \alpha * I_\text{background} + (1 - \alpha) * I_\text{new}$$.
    \item Stop the loop if the key q is pressed.
  \end{enumerate}
\end{itemize}

You can start by trying with $\alpha = 0.95$, T = 10, and A = 15000.

\subsubsection*{Exercise \theexno}
\addtocounter{exno}{1}

Implement and test the above program.

\subsubsection*{Exercise \theexno}
\addtocounter{exno}{1}

Try to change $\alpha$, T and A. What effects do it have?

\subsubsection*{Exercise \theexno}
\addtocounter{exno}{1}

The images are displayed using the OpenCV function \texttt{imshow}. The display window has several ways of zooming in the displayed image. One function is to zoom x30 that shows the pixel values as numbers. Do that and notice the change of the values.

\appendix
\section{Using a mobile phone camera}
\label{app:droidcam}

It is possible to use a mobile phone as a remote camera in OpenCV.

You need to install a web cam app on your phone. One option is \texttt{DroidCam} that can be installed from Google Play or from Apple App Store.

The computer and your phone should be on the same wireless network. For example one of the DTU wireless networks.

Now start the DroidCam application on your phone. It should now show an web-address, for example \url{http://192.168.1.120:4747/video}

Use this address, in the program:

\begin{verbatim}
use_droid_cam = True
if use_droid_cam:
    url = "http://192.168.1.120:4747/video"
cap = cv2.VideoCapture(url)
\end{verbatim}

You should now see the video from your mobile phone on your computer screen.


