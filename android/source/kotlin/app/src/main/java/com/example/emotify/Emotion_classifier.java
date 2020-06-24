package com.example.emotify;


import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class Emotion_classifier {
    private final Interpreter interpreter;

    private Emotion_classifier(Interpreter interpreter) {
        this.interpreter = interpreter;
    }

    public static Emotion_classifier classifier(AssetManager assetManager, String modelPath) throws IOException {
        ByteBuffer byteBuffer = loadModelFile(assetManager, modelPath);
        Interpreter interpreter = new Interpreter(byteBuffer);
        return new Emotion_classifier(interpreter);
    }

    private static ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public String recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        float[][] result = new float[1][Emotion.OUTPUT_LABELS.size()];
        interpreter.run(byteBuffer, result);
        return getMaxResult(result[0]);
    }

    private String getMaxResult(float[] result) {
        float probability = result[0];
        int index = 0;
        for (int i =0;i< 7;i++) {
            if (result[i] > probability) {
                probability = result[i];
                index = i;
            }
        }
        if(index==0) return "Anger";
        if(index==1) return "Disgust";
        if(index==2) return "Fear";
        if(index==3) return "Happy";
        if(index==4) return "Sad";
        if(index==5) return "Surprise";
        if(index==6) return "Neutral";

        return "Not known";

    }

    public ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(Emotion.MODEL_INPUT_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] pixels = new int[Emotion.INPUT_IMG_SIZE_WIDTH * Emotion.INPUT_IMG_SIZE_HEIGHT];
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

}
