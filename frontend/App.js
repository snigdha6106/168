import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Dimensions, SafeAreaView, Platform, Switch, TouchableOpacity } from 'react-native';
import { useState, useEffect, useRef } from 'react';
import { AILocalEngine } from './ai_inference';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import * as Location from 'expo-location';
import benchmarkStream from './assets/benchmark_imu_stream.json';

let MapView, Marker, Polyline, WebMap;
if (Platform.OS !== 'web') {
  const Maps = require('react-native-maps');
  MapView = Maps.default;
  Marker = Maps.Marker;
  Polyline = Maps.Polyline;
} else {
  WebMap = require('./WebMap').default;
}

const WS_URL = 'ws://192.168.29.65:8000/ws/telemetry';

export default function App() {
  const [telemetry, setTelemetry] = useState(null);
  const [path, setPath] = useState([]);
  const [truthPath, setTruthPath] = useState([]);
  const [connected, setConnected] = useState(false);
  
  const [inferenceMode, setInferenceMode] = useState('cloud');
  const modeRef = useRef('cloud');
  
  const [dataSource, setDataSource] = useState('benchmark');
  const dataRef = useRef('benchmark');
  
  const [isFollowing, setIsFollowing] = useState(true);
  const isFollowingRef = useRef(true);

  const mapRef = useRef(null);
  const aiEngine = useRef(null);
  const replayIndex = useRef(0);
  const latestSensors = useRef({ acc: {x:0, y:0, z:0}, gyro: {x:0, y:0, z:0} });

  const handleInteraction = () => {
    setIsFollowing(false);
    isFollowingRef.current = false;
  };

  const handleRecenter = () => {
    setIsFollowing(true);
    isFollowingRef.current = true;
    if (telemetry && Platform.OS !== 'web' && mapRef.current) {
      mapRef.current.animateToRegion({
        latitude: telemetry.estimated.lat,
        longitude: telemetry.estimated.lon,
        latitudeDelta: 0.005,
        longitudeDelta: 0.005,
      }, 500);
    }
  };

  useEffect(() => {
    if (!aiEngine.current) {
      aiEngine.current = new AILocalEngine();
    }
    let ws = new WebSocket(WS_URL);
    ws.onopen = () => { console.log('Connected'); setConnected(true); };
    ws.onmessage = (e) => {
      if (dataRef.current === 'live') return; // Ignore WS if in Live Sensor Mode
      try {
        const data = JSON.parse(e.data);
        if (modeRef.current === 'device' && data.measured.imu) {
            const localSpeed = aiEngine.current.predict(data.measured.imu);
            data.estimated.velocity = localSpeed;
            data.estimated.mode = "Edge AI (Network Stream)";
        }
        setTelemetry(data);
        updatePathsAndMap(data);
      } catch (err) { }
    };
    ws.onclose = () => { console.log('Disconnected'); setConnected(false); };
    return () => ws.close();
  }, []);

  // Offline Benchmark Replay
  useEffect(() => {
    let timer;
    if (!connected && modeRef.current === 'device' && dataRef.current === 'benchmark') {
      timer = setInterval(() => {
        if (replayIndex.current >= benchmarkStream.length) replayIndex.current = 0;
        const frame = benchmarkStream[replayIndex.current];
        const localSpeed = aiEngine.current.predict(frame.imu);
        
        const data = {
          gnss_active: false,
          estimated: { lat: frame.lat, lon: frame.lon, velocity: localSpeed, mode: "Edge AI (Offline Replay)" },
          truth: { lat: frame.lat, lon: frame.lon }
        };
        setTelemetry(data);
        updatePathsAndMap(data);
        replayIndex.current++;
      }, 100);
    }
    return () => clearInterval(timer);
  }, [connected, dataSource, inferenceMode]); // Need to re-trigger if mode changes while disconnected

  // Live Sensor Mode
  useEffect(() => {
    let accSub, gyroSub, locSub;
    if (dataSource === 'live') {
      (async () => {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
          alert('Permission to access location was denied');
          return;
        }
        
        Accelerometer.setUpdateInterval(100);
        Gyroscope.setUpdateInterval(100);
        
        accSub = Accelerometer.addListener(data => { latestSensors.current.acc = data; });
        gyroSub = Gyroscope.addListener(data => { latestSensors.current.gyro = data; });
        
        locSub = await Location.watchPositionAsync(
          { accuracy: Location.Accuracy.High, timeInterval: 1000, distanceInterval: 1 },
          (loc) => {
            const { acc, gyro } = latestSensors.current;
            let localSpeed = 0;
            let modeText = "Live Native GPS";
            
            if (modeRef.current === 'device') {
              const imuData = [acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z];
              localSpeed = aiEngine.current.predict(imuData);
              modeText = "Edge AI (Live Hardware IMU)";
            } else {
              localSpeed = loc.coords.speed || 0;
            }
            
            const data = {
              gnss_active: true,
              estimated: { lat: loc.coords.latitude, lon: loc.coords.longitude, velocity: localSpeed, mode: modeText },
              truth: { lat: loc.coords.latitude, lon: loc.coords.longitude }
            };
            setTelemetry(data);
            updatePathsAndMap(data);
          }
        );
      })();
    }
    return () => {
      if (accSub) accSub.remove();
      if (gyroSub) gyroSub.remove();
      if (locSub) locSub.remove();
    };
  }, [dataSource, inferenceMode]);

  const updatePathsAndMap = (data) => {
    const newEstimated = { latitude: data.estimated.lat, longitude: data.estimated.lon };
    setPath(prev => [...prev.slice(-100), newEstimated]);
    
    if (data.truth && data.truth.lat) {
        const newTruth = { latitude: data.truth.lat, longitude: data.truth.lon };
        setTruthPath(prev => [...prev.slice(-100), newTruth]);
    }

    if (isFollowingRef.current && mapRef.current && Platform.OS !== 'web') {
      mapRef.current.animateToRegion({
        latitude: newEstimated.latitude,
        longitude: newEstimated.longitude,
        latitudeDelta: 0.005,
        longitudeDelta: 0.005,
      }, 500);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={[styles.dashboard, telemetry && !telemetry.gnss_active ? styles.dashboardAlert : null]}>
        <Text style={styles.title}>Intelligent Dead Reckoning</Text>
        
        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>
             {inferenceMode === 'device' ? '🧠 On-Device Edge AI' : '☁️ Cloud / Server AI'}
          </Text>
          <Switch 
            value={inferenceMode === 'device'} 
            onValueChange={(val) => { const newMode = val ? 'device' : 'cloud'; setInferenceMode(newMode); modeRef.current = newMode; }}
            trackColor={{ false: "#767577", true: "#81b0ff" }}
            thumbColor={inferenceMode === 'device' ? "#2196F3" : "#f4f3f4"}
          />
        </View>

        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>
             {dataSource === 'live' ? '📡 Live Device Hardware' : '📼 Dataset Benchmark'}
          </Text>
          <Switch 
            value={dataSource === 'live'} 
            onValueChange={(val) => { const newSource = val ? 'live' : 'benchmark'; setDataSource(newSource); dataRef.current = newSource; }}
            trackColor={{ false: "#767577", true: "#4CAF50" }}
            thumbColor={dataSource === 'live' ? "#81C784" : "#f4f3f4"}
          />
        </View>
        
        <Text style={styles.statusText}>Server Connection: {connected ? '🟢 ONLINE' : '🔴 OFFLINE'}</Text>
        
        {telemetry && (
          <View style={styles.telemetryGrid}>
            <View style={styles.telemetryBox}>
              <Text style={styles.label}>SPEED</Text>
              <Text style={styles.value}>{(telemetry.estimated.velocity * 3.6).toFixed(1)} km/h</Text>
            </View>
            <View style={styles.telemetryBox}>
              <Text style={styles.label}>MODE</Text>
              <Text style={[styles.value, {color: telemetry.gnss_active || dataSource === 'live' ? '#4CAF50' : '#FF5252', fontSize: 13}]}>
                {telemetry.estimated.mode}
              </Text>
            </View>
          </View>
        )}
      </View>

      <View style={{flex: 1}}>
          {Platform.OS === 'web' ? (
            <View style={styles.webFallback}>
              <WebMap 
                path={path} 
                truthPath={truthPath} 
                currentPos={telemetry ? {latitude: telemetry.estimated.lat, longitude: telemetry.estimated.lon} : null}
                isFollowing={isFollowing}
                onInteraction={handleInteraction}
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
              onPanDrag={handleInteraction}
            >
              <Polyline coordinates={path} strokeColor="#2196F3" strokeWidth={5} />
              <Polyline coordinates={truthPath} strokeColor="#4CAF50" strokeWidth={3} lineDashPattern={[5, 5]} />
              
              {telemetry && (
                <Marker 
                  coordinate={{
                    latitude: telemetry.estimated.lat,
                    longitude: telemetry.estimated.lon
                  }}
                  title="Vehicle"
                >
                  <View style={styles.carMarker} />
                </Marker>
              )}
            </MapView>
          )}

          {!isFollowing && (
              <TouchableOpacity style={styles.recenterBtn} onPress={handleRecenter}>
                  <Text style={styles.recenterText}>📍 Recenter Camera</Text>
              </TouchableOpacity>
          )}
      </View>

      <View style={styles.legend}>
        <View style={styles.legendItem}><View style={[styles.colorBox, {backgroundColor: '#2196F3'}]}/><Text>Estimated Path</Text></View>
        <View style={styles.legendItem}><View style={[styles.colorBox, {backgroundColor: '#4CAF50'}]}/><Text>Ground Truth</Text></View>
      </View>
      
      <StatusBar style="light" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1E1E1E' },
  dashboard: { padding: 20, paddingTop: 40, backgroundColor: '#333', borderBottomWidth: 1, borderColor: '#444' },
  dashboardAlert: { backgroundColor: '#4a1111' },
  title: { fontSize: 20, fontWeight: 'bold', color: 'white', marginBottom: 10 },
  statusText: { color: '#aaa', fontSize: 12, marginBottom: 15 },
  switchRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 10, justifyContent: 'space-between' },
  switchLabel: { color: 'white', fontWeight: 'bold' },
  telemetryGrid: { flexDirection: 'row', justifyContent: 'space-between' },
  telemetryBox: { flex: 1, backgroundColor: '#222', padding: 15, borderRadius: 10, marginHorizontal: 5, alignItems: 'center' },
  label: { color: '#888', fontSize: 12, fontWeight: 'bold', marginBottom: 5 },
  value: { color: 'white', fontSize: 22, fontWeight: 'bold' },
  webFallback: { flex: 1, backgroundColor: '#fff' },
  map: { width: Dimensions.get('window').width, height: '100%' },
  carMarker: { width: 20, height: 20, backgroundColor: '#FFEB3B', borderRadius: 10, borderWidth: 2, borderColor: 'black' },
  legend: { flexDirection: 'row', justifyContent: 'space-around', padding: 10, backgroundColor: 'white' },
  legendItem: { flexDirection: 'row', alignItems: 'center' },
  colorBox: { width: 15, height: 15, marginRight: 5, borderRadius: 3 },
  recenterBtn: { position: 'absolute', bottom: 20, alignSelf: 'center', backgroundColor: '#333', paddingVertical: 10, paddingHorizontal: 20, borderRadius: 20, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.5, shadowRadius: 3 },
  recenterText: { color: 'white', fontWeight: 'bold' }
});
