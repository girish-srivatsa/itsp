package com.example.emotify;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

import static com.example.emotify.EmotionsModelConfig.CLASSIFICATION_THRESHOLD;
import static com.example.emotify.EmotionsModelConfig.MAX_CLASSIFICATION_RESULTS;

public class EmotionsClassifier {
    private final Interpreter interpreter;

    private EmotionsClassifier(Interpreter interpreter) {
        this.interpreter = interpreter;
    }

    public static EmotionsClassifier classifier(AssetManager assetManager, String modelPath) throws IOException {
        ByteBuffer byteBuffer = loadModelFile(assetManager, modelPath);
        Interpreter interpreter = new Interpreter(byteBuffer);
        return new EmotionsClassifier(interpreter);
    }

    private static ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public List<Classification> recognizeImage(Bitmap bitmap) {
       // Log.i("EmotionsClassifier","recognizeImage");
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        //Log.i("EmotionsClassifier","convertBitmapToByteBuffer crossed");
        float[][] result = new float[1][EmotionsModelConfig.OUTPUT_LABELS.size()];
       // Log.i("EmotionsClassifier","result float created");
        interpreter.run(byteBuffer, result);
        //Log.i("EmotionsClassifier","interpreter run cross sort start");
        return getSortedResult(result);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        //Log.i("EmotionsClassifier","convertBitmapToByteBuffer start");
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(EmotionsModelConfig.MODEL_INPUT_SIZE);
       // Log.i("EmotionsClassifier","convertBitmapToByteBuffer line57");
        byteBuffer.order(ByteOrder.nativeOrder());
        //Log.i("EmotionsClassifier","convertBitmapToByteBuffer line 59");
        int[] pixels = new int[EmotionsModelConfig.INPUT_IMG_SIZE_WIDTH * EmotionsModelConfig.INPUT_IMG_SIZE_HEIGHT];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int pixel : pixels) {
            float rChannel = (pixel >> 16) & 0xFF;
            float gChannel = (pixel >> 8) & 0xFF;
            float bChannel = (pixel) & 0xFF;
            float pixelValue = (rChannel + gChannel + bChannel) / 3 / 255.f;
            byteBuffer.putFloat(pixelValue);
        }
        return byteBuffer;
    }

    private List<Classification> getSortedResult(float[][] resultsArray) {
        //Log.i("EmotionsClassifier","getSortedResult start");
        PriorityQueue<Classification> sortedResults = new PriorityQueue<>(
                MAX_CLASSIFICATION_RESULTS,
                (lhs, rhs) -> Float.compare(rhs.confidence, lhs.confidence)
        );
        //Log.i("EmotionsClassifier","getSortedResult line 79");

        for (int i = 0; i < EmotionsModelConfig.OUTPUT_LABELS.size(); ++i) {
            float confidence = resultsArray[0][i];
            if (confidence > CLASSIFICATION_THRESHOLD) {
                EmotionsModelConfig.OUTPUT_LABELS.size();
                sortedResults.add(new Classification(EmotionsModelConfig.OUTPUT_LABELS.get(i), confidence));
            }
        }
        //Log.i("EmotionsClassifier","getSortedResult line 88");

        return new ArrayList<>(sortedResults);
    }
}