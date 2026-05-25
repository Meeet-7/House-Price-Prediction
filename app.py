import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")

# Load model and label encoders with error handling
model = None
label_encoders = None

try:
    if os.path.exists("mdl.joblib"):
        model = joblib.load("mdl.joblib")
        logger.info("✅ Model loaded successfully")
        logger.info(f"📊 Model expects {model.n_features_in_} features")
    else:
        logger.error("❌ Model file 'mdl.joblib' not found")
        
    if os.path.exists("fixed_label_encoders.joblib"):
        label_encoders = joblib.load("fixed_label_encoders.joblib")
        logger.info("✅ Label encoders loaded successfully")
    else:
        logger.error("❌ Label encoders file 'fixed_label_encoders.joblib' not found")
        
except Exception as e:
    logger.error(f"❌ Error loading model/encoders: {e}")

def build_complete_features(user_input):
    """Build complete feature vector matching the trained model (43 features)"""
    
    # Extract user inputs with defaults
    city = user_input.get("city", "Mumbai")
    property_type = user_input.get("property_type", "Apartment")
    bhk = int(user_input.get("bedrooms", 2))
    bathrooms = int(user_input.get("bathrooms", 1))
    area_sqft = float(user_input.get("area_sqft", 1000))
    age = int(user_input.get("age", 5))
    floor = int(user_input.get("floor", 1))
    total_floors = int(user_input.get("total_floors", 10))
    furnishing = user_input.get("furnishing", "Unfurnished")
    parking = user_input.get("parking", "No")
    facing = user_input.get("facing", "North")
    
    # Safe encoding function
    def safe_encode(encoder_name, value):
        if label_encoders and encoder_name in label_encoders:
            encoder = label_encoders[encoder_name]
            if hasattr(encoder, 'classes_') and value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                logger.warning(f"⚠️ Unknown {encoder_name}: {value}, using default")
                return 0
        return 0
    
    # Build feature vector based on typical dataset schema (43 features)
    features = []
    
    # Add all 43 features as before
    features.extend([
        safe_encode("City", city),  # 1
        0,  # 2. Locality
        safe_encode("Property_Type", property_type),  # 3
        0,  # 4. RERA_Approved
        bhk,  # 5. BHK
        bathrooms,  # 6. Bathrooms
        max(1, bhk - 1),  # 7. Balconies
        floor,  # 8. Floor
        age,  # 9. Age_of_Property_years
        0,  # 10. Ready_to_Move
        safe_encode("Furnishing", furnishing),  # 11
        safe_encode("Parking", parking),  # 12
        safe_encode("Facing", facing),  # 13
        1,  # 14. Gated_Community
        1 if total_floors > 3 else 0,  # 15. Lift_Available
        0,  # 16. Water_Supply
        1,  # 17. Security_Guard
        0,  # 18. Gym
        0,  # 19. Swimming_Pool
        1,  # 20. Power_Backup
        0,  # 21. Clubhouse
        1,  # 22. Play_Area
        2.5,  # 23. Near_School_km
        4.0,  # 24. Near_Hospital_km
        3.0,  # 25. Near_Metro_km
        1.5,  # 26. Near_Market_km
        area_sqft * 2.5,  # 27. Monthly_Maintenance
        1100.0,  # 28. EMI_Per_Lakh
        8.5,  # 29. Interest_Rate
        1,  # 30. Resale
        area_sqft * 12,  # 31. Property_Tax_Annual
        {"Mumbai": 80, "Delhi": 90, "Bengaluru": 60, "Chennai": 70, "Hyderabad": 65, "Kolkata": 75, "Pune": 55, "Ahmedabad": 85}.get(city, 65),  # 32. Pollution_Index
        50.0,  # 33. Noise_Index
        15.0,  # 34. Crime_Rate
        8.0,  # 35. Internet_Availability
        7.5,  # 36. Public_Transport_Score
        0,  # 37. Flood_Zone
        0,  # 38. Earthquake_Zone
        7.0,  # 39. Civic_Amenities_Rating
        6.5,  # 40. Market_Demand_Rating
        3.5,  # 41. Rental_Yield_Percent
        area_sqft,  # 42. Carpet_Area_sqft
        total_floors  # 43. Total_Floors
    ])
    
    logger.info(f"🔢 Built {len(features)} features for model")
    return np.array(features).reshape(1, -1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        logger.info(f"📩 Received data: {data}")

        if not model or not label_encoders:
            return jsonify({"error": "Model not loaded properly. Please check if model files are present."}), 500

        # Validate required fields
        city = data.get("city", "").strip()
        property_type = data.get("property_type", "").strip()
        
        if not city or not property_type:
            return jsonify({"error": "City and Property Type are required"}), 400

        # Validate numeric inputs
        try:
            area_sqft = float(data.get("area_sqft", 1000))
            if area_sqft <= 0:
                return jsonify({"error": "Area must be greater than 0"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid area value"}), 400

        logger.info(f"🏠 Processing: {city}, {property_type}, {area_sqft} sqft")

        # Build complete feature vector
        input_features = build_complete_features(data)
        
        if input_features is None:
            return jsonify({"error": "Failed to build feature vector"}), 400
            
        logger.info(f"🔢 Input features shape: {input_features.shape}")
        
        # Verify feature count matches model expectation
        if input_features.shape[1] != model.n_features_in_:
            return jsonify({
                "error": f"Feature mismatch: got {input_features.shape[1]}, expected {model.n_features_in_}"
            }), 400

        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Ensure prediction is positive and reasonable
        if prediction < 0:
            prediction = abs(prediction)
        
        # Cap unrealistic predictions
        if prediction > 10000:  # More than 1 crore
            prediction = prediction / 10
        
        # Format price in Indian currency
        price_lakhs = prediction
        price_formatted = f"₹ {price_lakhs:.2f} Lakhs"
        
        logger.info(f"💰 Predicted price: {price_formatted}")

        return jsonify({
            "price": price_formatted,
            "price_lakhs": round(price_lakhs, 2),
            "details": {
                "city": city,
                "property_type": property_type,
                "area_sqft": area_sqft,
                "features_count": input_features.shape[1]
            }
        })

    except Exception as e:
        logger.error(f"❌ Prediction Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "encoders_loaded": label_encoders is not None,
        "expected_features": model.n_features_in_ if model else 0,
        "python_version": sys.version,
        "files_present": {
            "mdl.joblib": os.path.exists("mdl.joblib"),
            "fixed_label_encoders.joblib": os.path.exists("fixed_label_encoders.joblib")
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
