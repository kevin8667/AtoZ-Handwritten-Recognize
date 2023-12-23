using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using static HandwrittenModel;

public class HandwrittenModel : MonoBehaviour
{
    [SerializeField]
    Texture2D _texture;

    [SerializeField]
    NNModel _model;

    private Model _runtimeModel;

    private IWorker _worker;

    [Serializable]
    public struct Prediction
    {
        // The most likely value for this prediction
        public int predictedValue;
        // The list of likelihoods for all the possible classes
        public float[] predicted;

        public void SetPrediction(Tensor t)
        {
            // Extract the float value outputs into the predicted array.
            predicted = t.AsFloats();
            // The most likely one is the predicted value.
            predictedValue = Array.IndexOf(predicted, predicted.Max());
            Debug.Log($"Predicted {predictedValue}");
        }
    }
    [SerializeField]
    Prediction _prediction;

    // Start is called before the first frame update
    void Start()
    {
        _runtimeModel = ModelLoader.Load(_model);

        _worker = WorkerFactory.CreateWorker(_runtimeModel, WorkerFactory.Device.GPU);

        _prediction = new Prediction();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // making a tensor out of a grayscale texture
            var channelCount = 1; //grayscale, 3 = color, 4 = color+alpha


            var inputX = new Tensor(_texture, channelCount);

            // Peek at the output tensor without copying it.
            Tensor outputY = _worker.Execute(inputX).PeekOutput();

            _prediction.SetPrediction(outputY);


            // Dispose of the input tensor manually (not garbage-collected).
            inputX.Dispose();
        }
    }

    private void OnDestroy()
    {
        // Dispose of the engine manually (not garbage-collected).
        _worker?.Dispose();
    }
}
