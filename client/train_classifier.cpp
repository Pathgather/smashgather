#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <algorithm>

#include "src/Classifier.hpp"
#include "src/Util.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, char* argv[]) {

  if (argc != 2 && argc != 3) {
    cout << "Usage: " << argv[0] << " train [validate]|test" << endl;
    cout << "Trains the classifier using data from training_data/* and writes the resulting SVM\n";
    cout << "to multi-class.svm. Adding validate will train the SVM on a subset of the samples and\n";
    cout << "test it against the remaining samples.\n";
    cout << "In test mode the same file is read and ran against all samples.\n";
    return 1;
  }

  string action(argv[1]);

  if (action == "train") {

    // Check if we want to do cross validation
    bool validate = argc > 2 && string(argv[2]) == "validate";

    // Get all the character names in the training data directory.
    auto characters = glob("./img/training_data/*");
    std::transform(characters.begin(), characters.end(), characters.begin(), (string (&)(string const&))basename);

    // Init the training data matrix and fill it with the samples while also adding the correct
    // labels to the labels matrix.
    Mat samples(0, 40*40*3, CV_32FC1);
    Mat labels(0, 1, CV_32S);

    for(auto &character : characters) {
      auto class_id = static_cast<double>(Classifier::class_id(character));

      if (class_id == -1) {
        cerr << "Unknown character: " << character << endl;
        return 1;
      }

      for(auto &filename : glob("./img/training_data/" + character + "/*.png")) {
        auto image = imread(filename);

        samples.push_back(Classifier::convert_image(image));
        labels.push_back(Mat(1, 1, CV_32S, {class_id}));
      }
    }

    auto train_data = TrainData::create(samples, ROW_SAMPLE, labels);

    // Set the train test split and shuffle the samples
    if (validate) {
      train_data->setTrainTestSplitRatio(0.9);
    }

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setC(1);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));

    cout << endl << "Training the classifier.. This will take a minute." << endl;

    svm->train(train_data);
    svm->save("multi-class.svm");

    cout << "Done." << endl;

    // Run the classifier on all the training samples and show the error
    if (validate) {
      cout << "Accuracy on test set: " << svm->calcError(train_data, true, noArray()) << "%" << endl;
    }

  } else if (action == "test") {

    Classifier classifier("multi-class.svm");

    for(auto &filename : glob("./img/training_data/*/*.png")) {
      auto character = classifier.classify(imread(filename));
      cout << filename << " " << character.name << endl;
    }

  } else {
    cout << "Unknown option: " << action << endl;
    return 1;
  }

  return 0;
}
