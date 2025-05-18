# Assignment 04 Compact Pattern

Sie sollen das Compact Pattern implementieren, welches alle Primzahlen des Input Arrays in ein kompaktes Output Array speichert. Als predicate-function können Sie diejenige der CPU-Implementierung auf die GPU portieren. Verwenden Sie dazu die präsentierten CUDA Themen, damit ihre Implementierung auch effizient und maximal parallel ausgeführt wird. Verwenden Sie das Compact-Pattern, welches in der Vorlesung vorgestellt wurde. 

### Folgende Vorgaben gelten: 
    Verwenden Sie die Struktur und Helper-Funktionen der bereits eigeführten CUDA Beispielen. 
    Es dürfen keine weiteren Libraries verwendet werden.

### Punkte werden vergeben: 

Implementierungen 
* 2P für eine korrekte GPU-Implementierung für 2048 Items (1 ThreadBlock) 
* 2P für eine korrekte GPU-Implementierung, welche bank-conflicts minimiert. 
* 3½P für eine korrekte GPU-Implementierung für alle Items (2048 ThreadBlocks) 
* 3½P für eine Implementation, welche 256-mal die Daten mittels compact neu ordnet. Pro Iteration/Ausführung soll ein anderes Datenset verwendet werden (ändern des Seeds). Die Implementation muss auf GPU Streams zurückgreifen, um weitere Parallelität zu erreichen. Die Daten (In- und Output) pro Iteration müssen nicht im Memory gehalten werden und können nach einer erfolgreichen Iteration verworfen/überschrieben werden. 

Dokumentation 
* ½P für die sinnvolle Untersuchungen/Messungen mit den CUDA Profiling Tools und deren Analyse. Ein sauberes Zeit-Profiling aller gewählten Ansätze (ebenso vs CPU Code). Unterscheiden Sie zwischen Init-Aufwand und der wiederholten Ausführung des Codes/der Kernel. 
*  ½P für eine saubere Dokumentation der verwendeten Ansätze und des allgemeinen Aufbaus des Codes. Sowie sinnvollen Schlussfolgerungen der erarbeiteten Resultate. 

**Falls keine oder eine unzureichende Dokumentation zum Code abgegeben wird, gehe ich davon aus, dass der Code abgeschrieben / mittels LLMs erstellt wurde. In diesem Fall werden keine Punkte vergeben.**

Die Abgabe erfolgt über einen Pull Request. Erstellen Sie am besten zu Beginn einen «dev» Branch, vom welchem Sie dann in den main einen Pull Request erstellen. Laden sie den Dozenten (Simon Marcin) als Reviewer in den Pull Request ein.   
