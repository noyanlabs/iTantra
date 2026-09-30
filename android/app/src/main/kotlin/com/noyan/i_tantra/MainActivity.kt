package com.noyan.i_tantra

import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.net.wifi.p2p.WifiP2pConfig
import android.net.wifi.p2p.WifiP2pDevice
import android.net.wifi.p2p.WifiP2pInfo
import android.net.wifi.p2p.WifiP2pManager
import android.os.Handler
import android.os.Looper
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.PrintWriter
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket

/** WiFi Direct (WifiP2pManager) + TCP line transport, and a loud non-interruptible alarm. */
@SuppressLint("MissingPermission")
class MainActivity : FlutterActivity() {
    private val port = 8988
    private val ui = Handler(Looper.getMainLooper())
    private var events: EventChannel.EventSink? = null

    private lateinit var p2p: WifiP2pManager
    private lateinit var channel: WifiP2pManager.Channel
    private var socket: Socket? = null
    private var server: ServerSocket? = null
    private var writer: PrintWriter? = null
    @Volatile private var session = 0
    @Volatile private var connecting = false
    private val sender = java.util.concurrent.Executors.newSingleThreadExecutor()

    private var player: MediaPlayer? = null
    private var focusReq: AudioFocusRequest? = null

    private fun emit(m: Map<String, Any?>) = ui.post { events?.success(m) }
    private fun log(msg: String) {
        android.util.Log.d("itantra", msg)
        emit(mapOf("e" to "log", "msg" to msg))
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        p2p = getSystemService(Context.WIFI_P2P_SERVICE) as WifiP2pManager
        channel = p2p.initialize(this, mainLooper, null)

        val f = IntentFilter().apply {
            addAction(WifiP2pManager.WIFI_P2P_PEERS_CHANGED_ACTION)
            addAction(WifiP2pManager.WIFI_P2P_CONNECTION_CHANGED_ACTION)
        }
        registerReceiver(object : BroadcastReceiver() {
            override fun onReceive(c: Context, i: Intent) {
                when (i.action) {
                    WifiP2pManager.WIFI_P2P_PEERS_CHANGED_ACTION -> p2p.requestPeers(channel) { list ->
                        emit(mapOf("e" to "peers", "peers" to list.deviceList.map {
                            mapOf("name" to it.deviceName, "address" to it.deviceAddress,
                                "connected" to (it.status == WifiP2pDevice.CONNECTED))
                        }))
                    }
                    WifiP2pManager.WIFI_P2P_CONNECTION_CHANGED_ACTION -> p2p.requestConnectionInfo(channel) { info ->
                        log("conn changed: formed=${info.groupFormed} owner=${info.isGroupOwner} addr=${info.groupOwnerAddress?.hostAddress}")
                        if (info.groupFormed) startSocket(info)
                        // formed=false also fires transiently while the group forms; only a live socket means a real drop
                        else if (socket != null) closeSocket("disconnected")
                    }
                }
            }
        }, f)

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, "itantra/p2p/events")
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(a: Any?, s: EventChannel.EventSink?) { events = s }
                override fun onCancel(a: Any?) { events = null }
            })

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "itantra/p2p").setMethodCallHandler { call, res ->
            when (call.method) {
                "discover" -> p2p.discoverPeers(channel, result(res))
                "connect" -> {
                    val cfg = WifiP2pConfig().apply { deviceAddress = call.argument<String>("address")!! }
                    emit(mapOf("e" to "state", "state" to "connecting"))
                    p2p.connect(channel, cfg, result(res))
                }
                "disconnect" -> { closeSocket("disconnected"); p2p.removeGroup(channel, result(res)) }
                "send" -> {
                    val line = call.argument<String>("line")!!
                    // Socket writes are forbidden on the main thread; single thread keeps message order.
                    sender.execute {
                        val ok = send(line)
                        ui.post { res.success(ok) }
                    }
                }
                "startAlarm" -> { startAlarm(); res.success(null) }
                "stopAlarm" -> { stopAlarm(); res.success(null) }
                else -> res.notImplemented()
            }
        }
    }

    private fun result(res: MethodChannel.Result) = object : WifiP2pManager.ActionListener {
        override fun onSuccess() = res.success(true)
        override fun onFailure(reason: Int) = res.success(false)
    }

    // ---- TCP transport: group owner listens, client connects (with retries) ----
    private fun startSocket(info: WifiP2pInfo) {
        if (socket != null || connecting) return // broadcasts fire repeatedly; set up once
        connecting = true
        val my = ++session
        Thread {
            try {
                val s: Socket
                if (info.isGroupOwner) {
                    server?.close()
                    server = ServerSocket().also {
                        it.reuseAddress = true
                        it.bind(InetSocketAddress(port))
                    }
                    log("owner: listening on $port")
                    s = server!!.accept()
                    log("owner: client accepted")
                } else {
                    var c: Socket? = null
                    var tries = 0
                    while (c == null && tries++ < 10 && my == session) {
                        try {
                            c = Socket().also { it.connect(InetSocketAddress(info.groupOwnerAddress, port), 3000) }
                        } catch (e: Exception) {
                            log("client: connect try $tries failed: ${e.message}")
                            Thread.sleep(500)
                        }
                    }
                    if (c == null) {
                        connecting = false
                        emit(mapOf("e" to "state", "state" to "disconnected"))
                        return@Thread
                    }
                    s = c
                }
                if (my != session) { log("socket setup superseded, dropping"); s.close(); return@Thread }
                s.tcpNoDelay = true
                socket = s
                connecting = false
                writer = PrintWriter(s.getOutputStream(), true)
                log("socket ready (owner=${info.isGroupOwner})")
                emit(mapOf("e" to "state", "state" to "connected"))
                val r = BufferedReader(InputStreamReader(s.getInputStream(), Charsets.UTF_8))
                while (my == session) {
                    val line = r.readLine() ?: run { log("peer closed socket"); null } ?: break
                    emit(mapOf("e" to "msg", "line" to line))
                }
            } catch (e: Exception) {
                log("socket error: ${e.message}")
            } finally {
                if (my == session) closeSocket("disconnected")
            }
        }.start()
    }

    private fun closeSocket(state: String) {
        session++
        connecting = false
        try { socket?.close() } catch (_: Exception) {}
        try { server?.close() } catch (_: Exception) {}
        socket = null; server = null; writer = null
        emit(mapOf("e" to "state", "state" to state))
    }

    private fun send(line: String): Boolean {
        val w = writer ?: return false
        return try { w.println(line); !w.checkError() } catch (e: Exception) { log("send failed: ${e.message}"); false }
    }

    // ---- Emergency alarm: max alarm volume, exclusive audio focus, looping ----
    private fun startAlarm() {
        if (player != null) return
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        am.setStreamVolume(AudioManager.STREAM_ALARM, am.getStreamMaxVolume(AudioManager.STREAM_ALARM), 0)
        val attrs = AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build()
        focusReq = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
            .setAudioAttributes(attrs).setOnAudioFocusChangeListener { }.build()
        am.requestAudioFocus(focusReq!!)
        val uri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
            ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)
        player = MediaPlayer().apply {
            setAudioAttributes(attrs)
            setDataSource(this@MainActivity, uri)
            isLooping = true
            prepare()
            start()
        }
    }

    private fun stopAlarm() {
        player?.run { try { stop() } catch (_: Exception) {}; release() }
        player = null
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        focusReq?.let { am.abandonAudioFocusRequest(it) }
        focusReq = null
    }
}
