package com.example.emotify

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private var emotionclassifier: Emotion_classifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        loadMnistClassifier()
    }

    val REQUEST_IMAGE_CAPTURE = 1

    fun dispatchTakePictureIntent(view: View) {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            takePictureIntent.resolveActivity(packageManager)?.also {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            val imageBitmap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(imageBitmap)
            process(imageBitmap)

        }
    }


    fun process(picture: Bitmap?) {
        val preprocessedImage = ImageUtils.prepareImageForClassification(picture)
        val recognitions =
            emotionclassifier!!.recognizeImage(preprocessedImage)
        emotion.text = recognitions.toString()
    }


    private fun loadMnistClassifier() {
        try {
            emotionclassifier = Emotion_classifier.classifier(assets, Emotion.MODEL_FILENAME)
        } catch (e: IOException) {
            Toast.makeText(
                this,
                "model couldn't be loaded. Check logs for details.",
                Toast.LENGTH_SHORT
            ).show()
            e.printStackTrace()
        }
    }


}
