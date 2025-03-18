document
  .getElementById("predictionForm")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    const formData = {
      lineID: document.getElementById("lineID").value,
      stationID: document.getElementById("stationID").value,
      deviceID: document.getElementById("deviceID").value,
      payType: document.getElementById("payType").value,
      time: document.getElementById("time").value.replace("T", " ") + ":00",
    };

    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          document.getElementById("result").innerHTML =
            "<strong>Error:</strong> " + data.error;
        } else {
          document.getElementById(
            "result"
          ).innerHTML = `<strong>Predicted Destination:</strong> ${
            data.predicted_destination
          } <br>
                         <strong>Prediction Confidence:</strong> ${(
                           data.predicted_class_probability * 100
                         ).toFixed(2)}%`;
        }
      })
      .catch((error) => {
        document.getElementById("result").innerHTML =
          "<strong>Error:</strong> " + error;
      });
  });





fetch("/congestion_alert")
  .then((response) => response.json())
  .then((data) => {
    if (data.peak_hours) {
      document.getElementById(
        "alert_message"
      ).innerHTML = `<strong>High congestion expected at:</strong> ${data.peak_hours.join(
        ", "
      )} hours.`;
    } else {
      document.getElementById("alert_message").innerHTML = data.message;
    }
  })
  .catch((error) => {
    document.getElementById("alert_message").innerHTML =
      "<strong>Error loading congestion data.</strong>";
  });





fetch("/passenger_flow")
  .then((response) => response.json())
  .then((data) => {
    let topStationsList = document.getElementById("top_stations_list");
    topStationsList.innerHTML = "";

    if (data.top_stations) {
      data.top_stations.forEach((station) => {
        let listItem = document.createElement("li");
        listItem.innerHTML = `<strong>Station ${station.stationID}:</strong> ${station.passenger_count} passengers`;
        topStationsList.appendChild(listItem);
      });
    } else {
      topStationsList.innerHTML = "<li>Error loading passenger data.</li>";
    }
  })
  .catch((error) => {
    document.getElementById("top_stations_list").innerHTML =
      "<li>Error loading passenger data.</li>";
  });




document
  .getElementById("route_form")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    let start = document.getElementById("start_station").value;
    let destination = document.getElementById("destination_station").value;

    fetch("/alternative_routes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start_station: start,
        destination_station: destination,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        let routeList = document.getElementById("route_suggestions");
        routeList.innerHTML = "";

        if (data.alternative_routes) {
          data.alternative_routes.forEach((route) => {
            let listItem = document.createElement("li");
            listItem.innerHTML = `<strong>Via Station ${route.via}:</strong> ${route.travel_time} min, Congestion: ${route.congestion_level}`;
            routeList.appendChild(listItem);
          });
        } else {
          routeList.innerHTML = "<li>No alternative routes found.</li>";
        }
      })
      .catch((error) => {
        document.getElementById("route_suggestions").innerHTML =
          "<li>Error finding routes.</li>";
      });
  });



  function calculatePeakFare() {
    let baseFare = document.getElementById("baseFare").value;
    let travelTime = document.getElementById("travelTime").value;
    
    fetch("/peak_fare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_fare: parseFloat(baseFare), travel_time: travelTime })
    })
    .then(response => response.json())
    .then(data => document.getElementById("peakFareResult").innerText = `Adjusted Fare: ₹${data.adjusted_fare} (${data.type})`)
    .catch(error => console.error("Error:", error));
}

function adjustStationFare() {
    let baseFare = document.getElementById("stationBaseFare").value.trim(); 
    let stationID = document.getElementById("stationIDval").value.trim();

    console.log("Base Fare:", baseFare);
    console.log("Station ID:", stationID);

    if (!baseFare || !stationID) {
        document.getElementById("stationFareResult").innerText = "Error: Please enter a valid fare and station ID.";
        return;
    }

    fetch("/station_fare_adjustment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            base_fare: parseFloat(baseFare), 
            station_id: parseInt(stationID)
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.adjusted_fare !== undefined) {
            document.getElementById("stationFareResult").innerText = `Adjusted Fare: ₹${data.adjusted_fare}`;
        } else {
            document.getElementById("stationFareResult").innerText = `Error: ${data.error}`;
        }
    })
    .catch(error => console.error("Error:", error));
}




function applyPaymentDiscount() {
    let baseFare = document.getElementById("paymentBaseFare").value;
    let payType = document.getElementById("paymentMethod").value;
    
    fetch("/payment_method_fare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_fare: parseFloat(baseFare), pay_type: payType })
    })
    .then(response => response.json())
    .then(data => document.getElementById("paymentDiscountResult").innerText = `Adjusted Fare: ₹${data.adjusted_fare} (Discount: ${data.discount_applied})`)
    .catch(error => console.error("Error:", error));
}


function fetchAndDisplayImage(apiUrl, imgElementId) {
    fetch(apiUrl)
    .then(response => response.json())
    .then(data => {
        if (data.image) {
            document.getElementById(imgElementId).src = "data:image/png;base64," + data.image;
        } else {
            document.getElementById(imgElementId).alt = "Error Loading Image";
        }
    })
    .catch(error => console.error("Error:", error));
}

// Fetch Images on Page Load
window.onload = function() {
    fetchAndDisplayImage("/congestion_plot", "congestionPlot");
    fetchAndDisplayImage("/peak_off_peak_plot", "peakOffPeakPlot");
    fetchAndDisplayImage("/od_matrix_mnl", "odMatrixMNL");
    fetchAndDisplayImage("/od_matrix_xgb", "odMatrixXGB");
};