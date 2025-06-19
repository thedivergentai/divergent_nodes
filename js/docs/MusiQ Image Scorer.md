# MusiQ Image Scorer

This node uses Google's MusiQ model to score images based on aesthetic and technical quality.

## Parameters

- **image** (`IMAGE`): The input image to be scored.
- **score_aesthetic** (`BOOLEAN`, default: `True`): Whether to calculate and return the aesthetic score.
- **score_technical** (`BOOLEAN`, default: `True`): Whether to calculate and return the technical score.
- **model_path** (`STRING`, optional): Path to a custom MusiQ model. Leave empty to use the default downloaded model.

## Outputs

- **AESTHETIC_SCORE** (`FLOAT`): The aesthetic quality score of the image (0.0 - 10.0).
- **TECHNICAL_SCORE** (`FLOAT`): The technical quality score of the image (0.0 - 100.0).
- **FINAL_AVERAGE_SCORE_OUT_OF_10** (`INT`): The rounded average of aesthetic and technical scores, scaled to 0-10.
- **FINAL_AVERAGE_SCORE_OUT_OF_100** (`INT`): The rounded average of aesthetic and technical scores, scaled to 0-100.
- **ERROR_MESSAGE** (`STRING`): Any error messages encountered during processing.

## Usage

Connect an image to the `image` input. You can choose to score aesthetic, technical, or both. The node will output the individual scores and a combined average score in two different scales.
