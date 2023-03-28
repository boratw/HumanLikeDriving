function RequestCurrentMap()
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentMap(xmlHttp.responseText);            
        }
    };
    
    xmlHttp.open("GET", "/v/curstate/" + current_step, true);
    xmlHttp.send();

    var xmlHttp2 = new XMLHttpRequest();
    xmlHttp2.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentExp(xmlHttp2.responseText);            
        }
    };
    
    xmlHttp2.open("GET", "/v/expinfo/", true);
    xmlHttp2.send();
}
function RequestLatentResult()
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleLatentResult(xmlHttp.responseText);            
        }
    };
    url = "/v/latentout/" + clicked + "/";
    for(var i = 0; i < 8; ++i)
        url += document.getElementById("value_l" + i).innerHTML + "/"
    xmlHttp.open("GET", url, true);
    xmlHttp.send();
}
function RequestCurrentAgent()
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentAgentResult(clicked, xmlHttp.responseText);            
        }
    };
    url = "/v/agentinfo/" + clicked + "/";
    xmlHttp.open("GET", url, true);
    xmlHttp.send();
}

function HandleCurrentMap(response)
{
    data = JSON.parse(response);
    vehicles = data["state"];
    routes = data["route"];
    DrawCanvas();
}

function HandleCurrentExp(response)
{
    data = JSON.parse(response);
    max_step = data["max_step"];
    latent_len = data["latent_len"];

    document.getElementById("slider_step").max = max_step;
    for(var i = 0; i < 8; ++i)
    {
        if(i >= latent_len)
            document.getElementById("div_l" + i).hidden = true;
    }
    labels =[]
    for(var i = 0; i < max_step; ++i)
    {
        labels.push(i)
    }

    linechart.data.labels = labels;

}
function HandleLatentResult(response)
{
    data = JSON.parse(response);
    latentoutput = data["predicted"];
    DrawCanvas();
}
function HandleCurrentAgentResult(c, response)
{
    j = JSON.parse(response);
    datalist = [];
    const bordercolors = ["#FF1493", "#DC143C", "#FF4500", "#FFA500", "#4B0082", "#4169E1", "#008B8B", "#006400"]
    for(var i = 0; i < 8; ++i)
    {
        datalist.push({
            label: (i < 4 ? "global" + i : "local" + (i-4)),
            data: [],
            borderColor: bordercolors[i],
            fill: false})
    }
    for(var i = 0; i < j["global_latent_record"].length; ++i)
    {
        datalist[0]["data"].push(j["global_latent_record"][i][0]);
        datalist[1]["data"].push(j["global_latent_record"][i][1]);
        datalist[2]["data"].push(j["global_latent_record"][i][2]);
        datalist[3]["data"].push(j["global_latent_record"][i][3]);
        datalist[4]["data"].push(j["local_latent_record"][i][0]);
        datalist[5]["data"].push(j["local_latent_record"][i][1]);
        datalist[6]["data"].push(j["local_latent_record"][i][2]);
        datalist[7]["data"].push(j["local_latent_record"][i][3]);
    }
    latent_data[c] = datalist;
    if(clicked == c)
    {
        linechart.data.datasets = datalist;
        linechart.update('none');
    }
}
