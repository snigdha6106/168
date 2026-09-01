import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Dimensions, SafeAreaView, Platform } from 'react-native';
import { useState, useEffect, useRef } from 'react';

let MapView, Marker, Polyline, WebMap;
if (Platform.OS !== 'web') {
  const Maps = require('react-native-maps');
  MapView = Maps.default;
  Marker = Maps.Marker;
  Polyline = Maps.Polyline;
} else {
  WebMap = require('./WebMap').default;
}

// For physical devices, you may need to replace 'localhost' with your computer's local IP (e.g., 192.168.1.x)
const WS_URL = 'ws://127.0.0.1:8000/ws/telemetry';

export default function App() {
  const [telemetry, setTelemetry] = useState(null);
  const [path, setPath] = useState([]);
  const [truthPath, setTruthPath] = useState([]);
  const [connected, setConnected] = useState(false);
  const mapRef = useRef(null);

  useEffect(() => {
    let ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('Connected to backend');
      setConnected(true);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setTelemetry(data);
        
        // Update paths
        const newEstimated = { latitude: data.estimated.lat, longitude: data.estimated.lon };
        setPath(prev => [...prev.slice(-100), newEstimated]); // Keep last 100 points
        
        const newTruth = { latitude: data.truth.lat, longitude: data.truth.lon };
        setTruthPath(prev => [...prev.slice(-100), newTruth]);

        // Animate map to follow car (Native only)
        if (mapRef.current && Platform.OS !== 'web') {
          mapRef.current.animateToRegion({
            latitude: newEstimated.latitude,
            longitude: newEstimated.longitude,
            latitudeDelta: 0.005,
            longitudeDelta: 0.005,
          }, 500);
        }
      } catch (err) {
        console.error("Error parsing telemetry", err);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected');
      setConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      {/* Top Dashboard */}
      <View style={[styles.dashboard, telemetry && !telemetry.gnss_active ? styles.dashboardAlert : null]}>
        <Text style={styles.title}>Intelligent Dead Reckoning</Text>
        <Text style={styles.statusText}>Connection: {connected ? '🟢 ONLINE' : '🔴 OFFLINE'}</Text>
        
        {telemetry && (
          <View style={styles.telemetryGrid}>
            <View style={styles.telemetryBox}>
              <Text style={styles.label}>SPEED</Text>
              <Text style={styles.value}>{(telemetry.estimated.velocity * 3.6).toFixed(1)} km/h</Text>
            </View>
            <View style={styles.telemetryBox}>
              <Text style={styles.label}>MODE</Text>
              <Text style={[styles.value, {color: telemetry.gnss_active ? '#4CAF50' : '#FF5252'}]}>
                {telemetry.estimated.mode}
              </Text>
            </View>
          </View>
        )}
      </View>

      {/* Map View */}
      {Platform.OS === 'web' ? (
        <View style={styles.webFallback}>
          <WebMap 
            path={path} 
            truthPath={truthPath} 
            currentPos={telemetry ? {latitude: telemetry.estimated.lat, longitude: telemetry.estimated.lon} : null} 
          />
        </View>
      ) : (
        <MapView 
          ref={mapRef}
          style={styles.map}
          initialRegion={{
            latitude: 28.6139,
            longitude: 77.2090,
            latitudeDelta: 0.05,
            longitudeDelta: 0.05,
          }}
        >
          {/* Estimated Path */}
          <Polyline coordinates={path} strokeColor="#2196F3" strokeWidth={5} />
          {/* Ground Truth Path (for comparison) */}
          <Polyline coordinates={truthPath} strokeColor="#4CAF50" strokeWidth={3} lineDashPattern={[5, 5]} />
          
          {telemetry && (
            <Marker 
              coordinate={{
                latitude: telemetry.estimated.lat,
                longitude: telemetry.estimated.lon
              }}
              title="Vehicle"
              description="Current Position"
            >
              <View style={styles.carMarker} />
            </Marker>
          )}
        </MapView>
      )}

      {/* Map Legend */}
      <View style={styles.legend}>
        <View style={styles.legendItem}><View style={[styles.colorBox, {backgroundColor: '#2196F3'}]}/><Text>Estimated Path</Text></View>
        <View style={styles.legendItem}><View style={[styles.colorBox, {backgroundColor: '#4CAF50'}]}/><Text>Ground Truth</Text></View>
      </View>
      
      <StatusBar style="light" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1E1E1E',
  },
  dashboard: {
    padding: 20,
    backgroundColor: '#333',
    borderBottomWidth: 1,
    borderColor: '#444',
  },
  dashboardAlert: {
    backgroundColor: '#502020', // Red tint when GNSS is lost
  },
  title: {
    color: '#fff',
    fontSize: 22,
    fontWeight: 'bold',
  },
  statusText: {
    color: '#aaa',
    marginBottom: 10,
  },
  telemetryGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  telemetryBox: {
    flex: 1,
  },
  label: {
    color: '#888',
    fontSize: 12,
    fontWeight: 'bold',
  },
  value: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  map: {
    flex: 1,
    width: Dimensions.get('window').width,
  },
  webFallback: {
    flex: 1,
    backgroundColor: '#fff'
  },
  carMarker: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#FFEB3B',
    borderWidth: 2,
    borderColor: '#000',
  },
  legend: {
    position: 'absolute',
    bottom: 30,
    left: 20,
    backgroundColor: 'rgba(255,255,255,0.9)',
    padding: 10,
    borderRadius: 8,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  colorBox: {
    width: 15,
    height: 15,
    marginRight: 10,
    borderRadius: 3,
  }
});
