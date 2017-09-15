//
//  ViewController.swift
//  GenderAge
//
//  Created by Daniele on 12/09/17.
//  Copyright © 2017 nexor. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    
    @IBOutlet weak var imgPicture: UIImageView?
    @IBOutlet weak var svInfo: UIStackView!
    @IBOutlet weak var lblAge: UILabel?
    @IBOutlet weak var lblGender: UILabel?
    
    enum PredictionType {
        case uigraphics
        case vision
    }
    
    var age: Age!
    var gender: Gender!
    
    let cameraMode = false
    let predictionType: PredictionType = .uigraphics
    
    let analyzing = "Analyzing..."
    let unknown = "Unknown"
    let error = "Error"
    let noPixels = "No pixels"
    let male = "Male"
    let female = "Female"
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        age = Age()
        gender = Gender()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func camera(_ sender: Any) {
        let cameraPicker = UIImagePickerController()
        cameraPicker.delegate = self
        
        if cameraMode {
            cameraPicker.sourceType = .camera
            cameraPicker.allowsEditing = true
            cameraPicker.cameraCaptureMode = .photo
            cameraPicker.cameraDevice = .rear
            cameraPicker.showsCameraControls = true
        }
        
        present(cameraPicker, animated: true)
    }
    
    //MARK: - UIImagePickerControllerDelegate
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        
        picker.dismiss(animated: true, completion: nil)
        
        guard let image = info["UIImagePickerControllerOriginalImage"] as? UIImage else {
            return
        }
        
        self.predictImage(image: image)
    }
    
    func predictImage(image: UIImage) {
    
        self.imgPicture?.image = image

        switch self.predictionType {
        case .uigraphics:
            self.predictImageUIGraphics(image: image)
        case .vision:
            self.predictImageVision(image: image)
        }
    }
    
    func predictImageVision(image: UIImage) {
        
        let ageModel = try! VNCoreMLModel(for: Age().model)
        let ageRequest = VNCoreMLRequest(model: ageModel) { (req, err) in
            
        }
        let ageHandler = VNImageRequestHandler(cgImage: image.cgImage!, options: [:])
        
        let genderModel = try! VNCoreMLModel(for: Gender().model)
        let genderRequest = VNCoreMLRequest(model: genderModel) { (req, err) in
            guard let results = req.results as? [VNCoreMLFeatureValueObservation], let firstResult = results.first else {
                self.resetOnError()
                return
            }
            //TODO: multiple results
            self.setGender(prob: firstResult.featureValue.multiArrayValue)
        }
        
        let handler = VNImageRequestHandler(cgImage: image.cgImage!, options: [:])
        try? handler.perform([ageRequest, genderRequest])
    }
    
    
    
    
    func myResultsMethod(request: VNRequest, error: Error?) {

    }
    
    
    func predictImageUIGraphics(image: UIImage) {
        
        let (pixelBuffer, _) = image.pixelBuffer()
        
        guard let pixels = pixelBuffer else {
            self.resetOnError()
            return
        }
        
        self.lblAge?.text = analyzing
        self.lblGender?.text = analyzing
        
        if let predAge = try? age.prediction(data: pixels) {
            self.lblAge?.text = "not yet implemented"
        }
        else {
            self.lblAge?.text = unknown
        }
        
        if let predGender = try? gender.prediction(data: pixels) {
            self.setGender(prob: predGender.prob)
        }
        else {
            self.lblGender?.text = unknown
        }
    }
    
    func resetOnError() {
        DispatchQueue.main.async {
            self.imgPicture?.image = nil
            self.lblAge?.text = self.noPixels
            self.lblGender?.text = self.noPixels
        }
    }
    
    func setGender(prob: MLMultiArray?) {
        DispatchQueue.main.async {
            guard let probArray = prob, probArray.count >= 2 else {
                self.lblGender?.text = self.error
                return
            }
            
            //0 => M
            //1 => F
            let m = probArray[0].doubleValue
            let f = probArray[1].doubleValue
            
            self.lblGender?.text = m > f ? self.male : self.female
        }
    }
    
}

