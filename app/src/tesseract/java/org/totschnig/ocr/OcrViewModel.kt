package org.totschnig.ocr

import Catalano.Imaging.FastBitmap
import Catalano.Imaging.Filters.BradleyLocalThreshold
import Catalano.Imaging.IApplyInPlace
import android.app.Application
import android.app.DownloadManager
import android.app.DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED
import android.content.Context
import android.content.SharedPreferences
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewModelScope
import androidx.preference.PreferenceManager
import com.googlecode.tesseract.android.TessBaseAPI
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import java.util.*
import androidx.core.net.toUri

import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.android.Utils

const val TESSERACT_DOWNLOAD_FOLDER = "tesseract4/fast/"

class OcrViewModel(application: Application) : BaseViewModel(application) {
    val prefKey = application.getString(R.string.pref_tesseract_language_key)

    val preferences: SharedPreferences
        get() = PreferenceManager.getDefaultSharedPreferences(getApplication())

    fun preprocessImage(image: Bitmap): Bitmap {
        val matImage = Mat()
        Utils.bitmapToMat(image, matImage)

        val scaledMat = Mat()
        Imgproc.resize(matImage, scaledMat, Size(), 2.0, 2.0, Imgproc.INTER_LINEAR)

        // 그레이스케일로 변환
        Imgproc.cvtColor(scaledMat, scaledMat, Imgproc.COLOR_RGB2GRAY)

        // 대비 향상 (Contrast adjustment)
        val equalizedMat = Mat()
        Imgproc.equalizeHist(scaledMat, equalizedMat)

        // 가우시안 블러
        val blurredMat = Mat()
        Imgproc.GaussianBlur(equalizedMat, blurredMat, Size(5.0, 5.0), 0.0)

        // 로컬 이진화 (Adaptive Thresholding)
        val adaptiveThresholdMat = Mat()
        Imgproc.adaptiveThreshold(
            blurredMat,  // 입력 이미지 (그레이스케일 및 대비 향상 처리된 이미지)
            adaptiveThresholdMat,  // 출력 이미지 (이진화)
            255.0,  // 최대값 (픽셀 값의 최대값)
            Imgproc.ADAPTIVE_THRESH_MEAN_C,  // 임계값 계산 방식
            Imgproc.THRESH_BINARY,  // 이진화 방식
            11,  // 블록 크기 (임계값 계산에 사용할 이웃 영역 크기)
            2.0  // 하위 임계값 조정 (블록 내 평균값에 추가할 값)
        )

        val resultBitmap = Bitmap.createBitmap(adaptiveThresholdMat.cols(), adaptiveThresholdMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(adaptiveThresholdMat, resultBitmap)

        return resultBitmap
    }

    fun runTextRecognition(uri: Uri) {
        viewModelScope.launch {
            withContext(Dispatchers.Default) {
                try {
                    val application = getApplication<Application>()
                    if (!tessDataExists(application)) {
                        throw IllegalStateException(application.getString(R.string.configuration_pending))
                    }

                    application.contentResolver.openInputStream(uri)
                        ?.use { inputStream ->
                            BitmapFactory.decodeStream(
                                inputStream,
                                null,
                                BitmapFactory.Options().apply {
                                    inMutable = true
                                    inPreferredConfig = Bitmap.Config.ARGB_8888
                                })
                        }?.let {
                            with(TessBaseAPI()) {
                                if (!init(
                                        File(
                                            application.getExternalFilesDir(null),
                                            TESSERACT_DOWNLOAD_FOLDER
                                        ).path, "eng+" + language()
                                    )
                                ) {
                                    throw IllegalStateException("Could not init Tesseract")
                                }
                                setVariable("tessedit_do_invert", TessBaseAPI.VAR_FALSE)
                                setVariable("load_system_dawg", TessBaseAPI.VAR_FALSE)
                                setVariable("load_freq_dawg", TessBaseAPI.VAR_FALSE)
                                setVariable("load_punc_dawg", TessBaseAPI.VAR_FALSE)
                                setVariable("load_number_dawg", TessBaseAPI.VAR_FALSE)
                                setVariable("load_unambig_dawg", TessBaseAPI.VAR_FALSE)
                                setVariable("load_bigram_dawg", TessBaseAPI.VAR_FALSE)
                                setVariable("load_fixed_length_dawgs", TessBaseAPI.VAR_FALSE)
                                pageSegMode = TessBaseAPI.PageSegMode.PSM_AUTO_OSD
                                /*
                                setImage(with(FastBitmap(it.rotate(getOrientation(uri)))) {
                                    toGrayscale()
                                    val g: IApplyInPlace = BradleyLocalThreshold()
                                    g.applyInPlace(this)
                                    toBitmap()
                                })
                                */
                                setImage(preprocessImage(FastBitmap(it.rotate(getOrientation(uri))).toBitmap()))
                                utF8Text
                                val lines = mutableListOf<Line>()
                                with(resultIterator) {
                                    begin()
                                    do {
                                        val lineText =
                                            getUTF8Text(TessBaseAPI.PageIteratorLevel.RIL_TEXTLINE)
                                        val lineBoundingRect =
                                            getBoundingRect(TessBaseAPI.PageIteratorLevel.RIL_TEXTLINE)
                                        val elements = mutableListOf<Element>()
                                        do {
                                            val wordText =
                                                getUTF8Text(TessBaseAPI.PageIteratorLevel.RIL_WORD)
                                            val wordBoundingRect =
                                                getBoundingRect(TessBaseAPI.PageIteratorLevel.RIL_WORD)
                                            elements.add(Element(wordText, wordBoundingRect))
                                        } while (!isAtFinalElement(
                                                TessBaseAPI.PageIteratorLevel.RIL_TEXTLINE,
                                                TessBaseAPI.PageIteratorLevel.RIL_WORD
                                            ) && next(TessBaseAPI.PageIteratorLevel.RIL_WORD)
                                        )
                                        lines.add(Line(lineText, lineBoundingRect, elements))
                                    } while (next(TessBaseAPI.PageIteratorLevel.RIL_TEXTLINE))
                                    delete()
                                    recycle()
                                    result.postValue(Result.success(Text(listOf(TextBlock(lines)))))
                                }
                            }
                        } ?: run {
                        result.postValue(Result.failure(Exception("Unable to open $uri")))
                    }
                } catch (e: Exception) {
                    Timber.e(e)
                    result.postValue(Result.failure(e))
                }
            }
        }
    }

    private fun Bitmap.rotate(degrees: Int): Bitmap =
        if (degrees == 0) this else Bitmap.createBitmap(
            this,
            0,
            0,
            width,
            height,
            Matrix().apply { postRotate(degrees.toFloat()) },
            true
        )

    private fun filePath(language: String) =
        "${TESSERACT_DOWNLOAD_FOLDER}tessdata/${language}.traineddata"

    private fun fileName(language: String) = "${language}.traineddata"

    fun tessDataExists(context: Context): Boolean = language()?.let {
        File(context.getExternalFilesDir(null), filePath(it)).exists()
    } == true

    private fun language(): String? {
        return preferences.getString(prefKey, null)
    }

    fun downloadTessData(context: Context) = language()?.let {
        val uri =
            //"https://github.com/tesseract-ocr/tessdata_fast/raw/4.0.0/${fileName(it)}".toUri()
            "https://github.com/tesseract-ocr/tessdata_best/raw/main/${fileName(it)}".toUri()
        ContextCompat.getSystemService(context, DownloadManager::class.java)?.enqueue(
            DownloadManager.Request(uri)
                .setTitle(
                    context.getString(R.string.pref_tesseract_language_title) + " : " +
                            getLanguages(context)[it]
                )
                .setDescription(it)
                .setDestinationInExternalFilesDir(context, null, filePath(it))
                .setNotificationVisibility(VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
        )
        getTesseractLanguageDisplayName(context, it)
    }

    fun getLanguages(context: Context): Map<String, String> =
        context.resources.getStringArray(R.array.pref_tesseract_language_values)
            .map { it to getTesseractLanguageDisplayName(context, it) }
            .sortedBy { it.second }
            .toMap()

    private fun getTesseractLanguageDisplayName(context: Context, localeString: String): String {
        val localeParts = localeString.split("_")
        val lang = when (localeParts[0]) {
            "kmr" -> "kur"
            "chi" -> "zho"
            else -> localeParts[0]
        }
        val localeFromContext = context.resources.configuration.locale
        return if (localeParts.size == 2) {
            val script = when (localeParts[1]) {
                "sim" -> "Hans"
                "tra" -> "Hant"
                else -> localeParts[1]
            }
            Locale.Builder().setLanguage(lang).setScript(script).build().getDisplayName(
                localeFromContext
            )
        } else
            Locale(lang).getDisplayName(localeFromContext)
    }
}