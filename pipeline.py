import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import subprocess
import json
import traceback


# ============================================================
# NEUROSIGHT AI
# BRAIN MRI ANALYSIS PIPELINE
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "classifier",
    "tumor_classifier.keras"
)

LABELS = [
    "No Tumor",
    "Glioma",
    "Meningioma",
    "Pituitary"
]


# ============================================================
# UTILITY
# ============================================================

def run_script(script_path, *args):

    if not os.path.exists(script_path):
        return {
            "success": False,
            "error": f"Script not found: {script_path}"
        }

    try:

        result = subprocess.run(
            [
                sys.executable,
                script_path,
                *args
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_mri(image_path):

    results = {
        "success": False,
        "image": image_path,
        "classification": {},
        "segmentation": {},
        "radiomics": {},
        "gradcam": {},
        "errors": []
    }

    try:

        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if not os.path.exists(image_path):

            results["errors"].append(
                f"Image not found: {image_path}"
            )

            return results


        base_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]


        print("\n" + "=" * 65)
        print("                 NEUROSIGHT AI")
        print("             BRAIN MRI ANALYSIS")
        print("=" * 65)

        print("\nInput MRI:")
        print(image_path)


        # ====================================================
        # STEP 1 - CNN CLASSIFICATION
        # ====================================================

        print("\n" + "=" * 65)
        print("STEP 1/4 - CNN TUMOR CLASSIFICATION")
        print("=" * 65)

        if not os.path.exists(MODEL_PATH):

            results["errors"].append(
                f"Classifier model not found: {MODEL_PATH}"
            )

            return results

        print("\nLoading classifier...")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )


        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            results["errors"].append(
                "OpenCV could not read the MRI image."
            )

            return results


        # Resize exactly as model expects
        image = cv2.resize(
            image,
            (128, 128)
        )

        image = image.astype(
            np.float32
        ) / 255.0


        # Add channel dimension
        image = np.expand_dims(
            image,
            axis=-1
        )


        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )


        print("Running CNN...")

        prediction = model.predict(
            image,
            verbose=0
        )


        predicted_class = int(
            np.argmax(prediction[0])
        )

        confidence = float(
            prediction[0][predicted_class]
        ) * 100


        # Prevent label index error
        if predicted_class < len(LABELS):

            diagnosis = LABELS[
                predicted_class
            ]

        else:

            diagnosis = "Unknown"


        probabilities = {}

        for i, label in enumerate(LABELS):

            if i < len(prediction[0]):

                probabilities[label] = round(
                    float(prediction[0][i]) * 100,
                    2
                )


        results["classification"] = {

            "diagnosis": diagnosis,

            "confidence": round(
                confidence,
                2
            ),

            "probabilities": probabilities
        }


        print("\nCLASSIFICATION RESULT")
        print("-" * 50)

        print(
            f"Prediction : {diagnosis}"
        )

        print(
            f"Confidence : {confidence:.2f}%"
        )


        # ====================================================
        # STEP 2 - U-NET SEGMENTATION
        # ====================================================

        print("\n" + "=" * 65)
        print("STEP 2/4 - U-NET TUMOR SEGMENTATION")
        print("=" * 65)


        segmentation_script = os.path.join(
            PROJECT_ROOT,
            "models",
            "segmentation",
            "predict_mask.py"
        )


        segmentation_result = run_script(
            segmentation_script,
            image_path
        )


        if segmentation_result["success"]:

            print("U-Net segmentation completed.")

            mask_path = os.path.join(
                PROJECT_ROOT,
                "reports",
                "segmentation",
                f"{base_name}_predicted_mask.png"
            )


            # Check whether mask was actually generated

            if os.path.exists(mask_path):

                results["segmentation"] = {

                    "success": True,

                    "mask": mask_path
                }

            else:

                results["segmentation"] = {

                    "success": False,

                    "error":
                    "U-Net finished but mask file was not generated."
                }

                results["errors"].append(
                    "Segmentation mask was not generated."
                )

        else:

            print("WARNING: U-Net failed.")

            results["segmentation"] = {

                "success": False,

                "error":
                segmentation_result.get(
                    "stderr",
                    "Unknown segmentation error"
                )
            }

            results["errors"].append(
                "U-Net segmentation failed."
            )


        # ====================================================
        # STEP 3 - QUANTITATIVE ANALYSIS
        # ====================================================

        print("\n" + "=" * 65)
        print("STEP 3/4 - TUMOR QUANTITATIVE ANALYSIS")
        print("=" * 65)


        if (
            results["segmentation"].get("success")
            and os.path.exists(mask_path)
        ):

            features_script = os.path.join(
                PROJECT_ROOT,
                "utils",
                "tumor_features.py"
            )


            features_result = run_script(
                features_script,
                mask_path
            )


            if features_result["success"]:

                print(
                    "Tumor quantitative analysis completed."
                )

                results["radiomics"] = {

                    "success": True,

                    "output":
                    features_result["stdout"]
                }

            else:

                print(
                    "WARNING: Quantitative analysis failed."
                )

                results["radiomics"] = {

                    "success": False,

                    "error":
                    features_result.get(
                        "stderr",
                        "Unknown radiomics error"
                    )
                }

                results["errors"].append(
                    "Tumor quantitative analysis failed."
                )

        else:

            print(
                "Skipping quantitative analysis because mask is unavailable."
            )

            results["radiomics"] = {

                "success": False,

                "error":
                "Segmentation mask unavailable."
            }


        # ====================================================
        # STEP 4 - GRAD-CAM
        # ====================================================

        print("\n" + "=" * 65)
        print("STEP 4/4 - GRAD-CAM EXPLAINABILITY")
        print("=" * 65)


        gradcam_script = os.path.join(
            PROJECT_ROOT,
            "models",
            "classifier",
            "gradcam.py"
        )


        gradcam_result = run_script(
            gradcam_script,
            image_path
        )


        gradcam_path = os.path.join(
            PROJECT_ROOT,
            "reports",
            "explainability",
            f"{base_name}_gradcam_result.png"
        )


        if gradcam_result["success"]:

            print(
                "Grad-CAM completed."
            )


            if os.path.exists(gradcam_path):

                results["gradcam"] = {

                    "success": True,

                    "image": gradcam_path
                }

            else:

                results["gradcam"] = {

                    "success": True,

                    "image": None,

                    "warning":
                    "Grad-CAM completed but output image was not found."
                }

        else:

            print(
                "WARNING: Grad-CAM failed."
            )

            results["gradcam"] = {

                "success": False,

                "error":
                gradcam_result.get(
                    "stderr",
                    "Unknown Grad-CAM error"
                )
            }

            results["errors"].append(
                "Grad-CAM failed."
            )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        results["success"] = True


        print("\n" + "=" * 65)
        print("              NEUROSIGHT AI COMPLETE")
        print("=" * 65)


        print("\nFINAL ANALYSIS")
        print("-" * 65)

        print(
            f"Diagnosis       : {diagnosis}"
        )

        print(
            f"Confidence      : {confidence:.2f}%"
        )

        print(
            "Segmentation    :",
            "Completed"
            if results["segmentation"].get("success")
            else "Failed"
        )

        print(
            "Tumor Analysis  :",
            "Completed"
            if results["radiomics"].get("success")
            else "Skipped / Failed"
        )

        print(
            "Grad-CAM        :",
            "Completed"
            if results["gradcam"].get("success")
            else "Failed"
        )


        print("\n" + "=" * 65)


        return results


    except Exception as e:

        print("\nPIPELINE ERROR")
        print(str(e))

        traceback.print_exc()

        results["success"] = False

        results["errors"].append(
            str(e)
        )

        return results


# ============================================================
# COMMAND LINE MODE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python pipeline.py path/to/mri.png"
        )

        sys.exit(1)


    image_path = sys.argv[1]


    if not os.path.isabs(image_path):

        image_path = os.path.abspath(
            os.path.join(
                PROJECT_ROOT,
                image_path
            )
        )


    results = analyze_mri(
        image_path
    )


    # Print JSON result for Flask/frontend

    print("\nPIPELINE_JSON_START")

    print(
        json.dumps(
            results,
            indent=2,
            default=str
        )
    )

    print("PIPELINE_JSON_END")